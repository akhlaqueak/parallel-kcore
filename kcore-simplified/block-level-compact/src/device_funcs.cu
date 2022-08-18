
#include "../inc/device_funcs.h"
#include "stdio.h"
#include "buffer.cc"

__device__ void scanBlock(unsigned int* addresses){
    // Hillis Steele Scan
    // todo check this code is working
    __syncthreads();
    int initVal = addresses[THID];

    for (unsigned int d = 1; d < BLK_DIM; d = d*2) {
        unsigned int newVal = addresses[THID];   
        if (int(THID - d) >= 0)  
            newVal += addresses[THID-d];  
        __syncthreads();
        addresses[THID] = newVal;
        __syncthreads();  
    }
        //Hillis-Steele Scan gives inclusive scan.
        //to get exclusive scan, subtract the initial values.
    addresses[THID] -= initVal;
    __syncthreads();  
}

__device__ void selectNodesAtLevel(unsigned int *degrees, unsigned int V, unsigned int* shBuffer, unsigned int* glBuffer, unsigned int* bufTailPtr, unsigned int level){

    unsigned int glThreadIdx = blockIdx.x * BLK_DIM + THID; 

    __shared__ bool predicate[BLK_DIM];
    __shared__ unsigned int addresses[BLK_DIM];
    __shared__ unsigned int bTail;
    
    for(unsigned int base = 0; base < V; base += N_THREADS){
        
        unsigned int v = base + glThreadIdx; 

        // all threads should get some value, if vertices are less than n_threads, rest of the threads get zero
        predicate[THID] = (v<V)? (degrees[v] == level) : 0;

        addresses[THID] = predicate[THID];

        scanBlock(addresses);

        
        if(THID == BLK_DIM - 1){  
            int nv =  addresses[THID] + predicate[THID];            
            bTail = nv>0? atomicAdd(bufTailPtr, nv) : 0;
            
        }

        // this sync is necessary so that memory is allocated before writing to buffer
        __syncthreads();
        
        addresses[THID] += bTail;
        
        if(predicate[THID])
            writeToBuffer(shBuffer, glBuffer, addresses[THID], v);
        
        __syncthreads();
            
    }
}




__device__ void syncBlocks(unsigned int* blockCounter){
    
    if (THID==0)
    {
        atomicAdd(blockCounter, 1);
        __threadfence();
        while(ldg(blockCounter)<BLK_NUMS){
            // number of blocks can't be greater than SMs, else it'll cause infinite loop... 
            // printf("%d ", blockCounter[0]);
        };// busy wait until all blocks increment
    }   
    __syncthreads();
}

__global__ void PKC(G_pointers d_p, unsigned int *global_count, int level, int V, 
                    unsigned int* blockCounter, unsigned int* glBuffers){


    __shared__ unsigned int shBuffer[MAX_NV];
    __shared__ unsigned int bufTail;
    __shared__ unsigned int* glBuffer;
    __shared__ unsigned int base;
    __shared__ unsigned int lock;
    unsigned int warp_id = THID / 32;
    unsigned int lane_id = THID % 32;
    unsigned int i;

    bufTail = 0;
    glBuffer = NULL;
    base = 0;
    lock = 0;
    // glBuffer = glBuffers + blockIdx.x * GLBUFFER_SIZE; 

    __syncthreads();

    selectNodesAtLevel(d_p.degrees, V, shBuffer, glBuffers + blockIdx.x * GLBUFFER_SIZE, &bufTail, level);

    syncBlocks(blockCounter);

    // bufTail is being incrmented within the loop, 
    // warps should process all the nodes added during the execution of loop
    // for that purpose e_processes is introduced, is incremented whenever a warp takes a job. 
    
    
    // for(unsigned int i = warp_id; i<bufTail ; i = warp_id + base){
    // this for loop is a wrong choice, as many threads will exit from the loop checking the condition
    while(true){
        __syncthreads(); //syncthreads must be executed by all the threads, so can't put after break or continue...

        if(base == bufTail) break;

        i = base + warp_id;
        
        if(THID == 0){
            base += WARPS_EACH_BLK;
            if(bufTail < base )
                base = bufTail;
        }
        __syncthreads();
        if(i >= bufTail) continue; // this warp won't have to do anything     
        
        
        unsigned int v, start, end;

        v = readFromBuffer(shBuffer, glBuffers + blockIdx.x * GLBUFFER_SIZE, i);
        start = d_p.neighbors_offset[v];
        end = d_p.neighbors_offset[v+1];


        while(true){
            __syncwarp();

            if(start >= end) break;

            unsigned int j = start + lane_id;
            start += WARP_SIZE;
            if(j >= end) continue;

            unsigned int u = d_p.neighbors[j];
            if(ldg(d_p.degrees+u) > level){
                unsigned int a = atomicSub(d_p.degrees+u, 1);
            
                if(a == level+1){
                    unsigned int loc = atomicAdd(&bufTail, 1);
                    writeToBuffer(shBuffer, glBuffers + blockIdx.x * GLBUFFER_SIZE, loc, u);
                }

                if(a <= level){
                    // node degree became less than the level after decrementing... 
                    atomicAdd(d_p.degrees+u, 1);
                }
            }
        }

    }


    if(THID == 0 ){
        if(bufTail>0) atomicAdd(global_count, bufTail); // atomic since contention among blocks
        if(glBuffer!=NULL) free(glBuffer);
    }

}


