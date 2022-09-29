
#include "../inc/device_funcs.h"
#include "stdio.h"
#include "buffer.cc"
#define SHBUFFER 1

__global__ void selectNodesAtLevel(unsigned int *degrees, unsigned int level, unsigned int V, 
                 unsigned int* bufTails, unsigned int* glBuffers){

    __shared__ unsigned int* glBuffer; 
    __shared__ unsigned int bufTail; 
    
    if(THID == 0){
        bufTail = 0;
        glBuffer = glBuffers + blockIdx.x*GLBUFFER_SIZE;
    }
    __syncthreads();

    unsigned int global_threadIdx = blockIdx.x * blockDim.x + threadIdx.x; 
    for(unsigned int base = 0; base < V; base += N_THREADS){
        
        unsigned int v = base + global_threadIdx; 

        if(v >= V) continue;

        if(degrees[v] == level){
            unsigned int loc = atomicAdd(&bufTail, 1);
            writeToBuffer(glBuffer, loc, v);
        }
    }

    __syncthreads();

    if(THID == 0) 
    {
        bufTails [blockIdx.x] = bufTail;
    }
}




__global__ void processNodes(G_pointers d_p, int level, int V, 
                    unsigned int* bufTails, unsigned int* glBuffers, 
                    unsigned int *global_count){
#ifdef SHBUFFER
    __shared__ unsigned int shBuffer[MAX_NV];
#endif
    __shared__ unsigned int bufTail;
    __shared__ unsigned int* glBuffer;
    __shared__ unsigned int base;
    unsigned int warp_id = THID / 32;
    unsigned int lane_id = THID % 32;
    unsigned int regTail;
    unsigned int i, initTail;
    if(THID==0){
        bufTail = bufTails[blockIdx.x];
        base = 0;
        initTail = bufTail;
        glBuffer = glBuffers + blockIdx.x*GLBUFFER_SIZE; 
        assert(glBuffer!=NULL);
    }

    // bufTail is being incrmented within the loop, 
    // warps should process all the nodes added during the execution of loop
    
    // for(unsigned int i = warp_id; i<bufTail ; i +=warps_each_block ){
    // this for loop is a wrong choice, as many threads will exit from the loop checking the condition
    while(true){
        __syncthreads(); //syncthreads must be executed by all the threads
        if(base == bufTail) break; // all the threads will evaluate to true at same iteration
        i = base + warp_id;
        regTail = bufTail;
        __syncthreads();

        if(i >= regTail) continue; // this warp won't have to do anything            

        if(THID == 0){
            // base += min(WARPS_EACH_BLK, regTail-base)
            // update base for next iteration
            base += WARPS_EACH_BLK;
            if(regTail < base )
                base = regTail;
        }
        //bufTail is incremented in the code below:

        unsigned int v = readFromBuffer(glBuffer, i);
        unsigned int start = d_p.neighbors_offset[v];
        unsigned int end = d_p.neighbors_offset[v+1];


        while(true){
            __syncwarp();

            if(start >= end) break;

            unsigned int j = start + lane_id;
            start += WARP_SIZE;
            if(j >= end) continue;

            unsigned int u = d_p.neighbors[j];
            if(*(d_p.degrees+u) > level){
                
                unsigned int a = atomicSub(d_p.degrees+u, 1);
            
                if(a == level+1){
                    unsigned int loc = atomicAdd(&bufTail, 1);
                    #if SHBUFFER
                        writeToBuffer(shBuffer, glBuffer, initTail, loc, u);
                    #else
                        writeToBuffer(glBuffer, loc, u);
                    #endif
                }

                if(a <= level){
                    // node degree became less than the level after decrementing... 
                    atomicAdd(d_p.degrees+u, 1);
                }
            }
        }

    }

    if(THID == 0 && bufTail>0){
        atomicAdd(global_count, bufTail); // atomic since contention among blocks
    }
}
