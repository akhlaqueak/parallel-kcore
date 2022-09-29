
#include "../inc/device_funcs.h"
#include "stdio.h"
#include "scans.cc"

__device__ inline void writeToBuffer(unsigned int* glBuffer, unsigned int loc, unsigned int v){
    assert(loc < GLBUFFER_SIZE);
    glBuffer[loc] = v;
}


__device__ inline unsigned int readFromBuffer(unsigned int* glBuffer, unsigned int loc){
    assert(loc < GLBUFFER_SIZE);
    return glBuffer[MAX_NV]; 
}


__global__ void selectNodesAtLevel(unsigned int *degrees, unsigned int level, unsigned int V, 
    unsigned int* bufTails, unsigned int* glBuffers){

    __shared__ volatile unsigned int addresses[BLK_DIM];
    __shared__ bool predicate[BLK_DIM];
    __shared__ unsigned int temp[BLK_DIM];

    __shared__ unsigned int* glBuffer; 
    __shared__ unsigned int bufTail; 
    
    if(THID == 0){
        bufTail = 0;
        glBuffer = glBuffers + blockIdx.x*GLBUFFER_SIZE;
    }
    __syncthreads();

    unsigned int glThreadIdx = blockIdx.x * BLK_DIM + THID; 

    for(unsigned int base = 0; base < V; base += N_THREADS){
        
        unsigned int v = base + glThreadIdx; 

        // all threads should get some value, if vertices are less than n_threads, rest of the threads get zero
        predicate[THID] = (v<V)? (degrees[v] == level) : 0;
        temp[THID] = v;
        
        // either select block level compact or warp compact

        // compactBlock(predicate, addresses, temp, glBuffer, &bufTail); 
        compactWarp(predicate, addresses, temp, glBuffer, &bufTail);       
        __syncthreads();            
    }

    if(THID == 0)
        bufTails[blockIdx.x] = bufTail;
}




__global__ void processNodes(G_pointers d_p, int level, int V, 
    unsigned int* bufTails, unsigned int* glBuffers, 
    unsigned int *global_count){
        
    __shared__ volatile unsigned int addresses[BLK_DIM];
    __shared__ bool predicate[BLK_DIM];
    __shared__ unsigned int temp[BLK_DIM];
    __shared__ unsigned int bufTail;
    __shared__ unsigned int base;
    __shared__ unsigned int* glBuffer;
    unsigned int warp_id = THID / 32;
    unsigned int lane_id = THID % 32;
    unsigned int i, regTail;
    
    if(THID==0){
        base = 0; 
        glBuffer = glBuffers + blockIdx.x * GLBUFFER_SIZE; 
        bufTail = bufTails[blockIdx.x];
    }

    predicate[THID] = 0;

    
    // bufTail is being incrmented within the loop, 
    // warps should process all the nodes added during the execution of loop
    // for that purpose e_processes is introduced, is incremented whenever a warp takes a job. 
    
    
    // for(unsigned int i = warp_id; i<bufTail ; i = warp_id + base){
    // this for loop is a wrong choice, as many threads will exit from the loop checking the condition
    while(true){
        __syncthreads(); //syncthreads must be executed by all the threads
        if(base == bufTail) break;
        i = base + warp_id;
        regTail = bufTail;        
        __syncthreads(); // this call is necessary, so that following update to base is done after everyone get value of i

        
        if(THID == 0){
            base += WARPS_EACH_BLK;
            if(regTail<base)
            base = regTail;
            // base += min(WARPS_EACH_BLK, regTail-base);
        }     
        if(i >= regTail) continue; // this warp won't have to do anything     
        
        unsigned int v = readFromBuffer(glBuffer, i);
        unsigned int start = d_p.neighbors_offset[v];
        unsigned int end = d_p.neighbors_offset[v+1];


        while(true){
            // __syncwarp();

            compactWarp(predicate, addresses, temp, glBuffer, &bufTail);
            if(start >= end) break;

            unsigned int j = start + lane_id;
            start += WARP_SIZE;
            if(j >= end) continue;

            unsigned int u = d_p.neighbors[j];
            if(ldg(d_p.degrees+u) > level){
                unsigned int a = atomicSub(d_p.degrees+u, 1);
            
                if(a == level+1){
                    temp[THID] = u;
                    predicate[THID] = 1;
                    // unsigned int loc = atomicAdd(&bufTail, 1);
                    // writeToBuffer(shBuffer, glBuffer, loc, u);
                }

                if(a <= level){
                    // node degree became less than the level after decrementing... 
                    atomicAdd(d_p.degrees+u, 1);
                }
            }
        }

    }
    __syncthreads();

    if(THID == 0 ){
        if(bufTail>0) atomicAdd(global_count, bufTail); // atomic since contention among blocks
        // if(glBuffer!=NULL) free(glBuffer);
    }

}



