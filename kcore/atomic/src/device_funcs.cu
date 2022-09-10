
#include "../inc/device_funcs.h"
#include "stdio.h"
#include "buffer.cc"

__device__ unsigned long long int ct = 0;

__device__ void selectNodesAtLevel(unsigned int *degrees, unsigned int V, unsigned int* shBuffer, unsigned int* glBuffer, unsigned int* bufTail, unsigned int level){
    unsigned int global_threadIdx = blockIdx.x * blockDim.x + threadIdx.x; 
    for(unsigned int base = 0; base < V; base += N_THREADS){
        
        unsigned int v = base + global_threadIdx; 

        if(v >= V) continue;

        if(degrees[v] == level){
            unsigned int loc = atomicAdd(bufTail, 1);
            writeToBuffer(shBuffer, glBuffer, loc, v);
        }
    }
    if(THID==0)
        atomicAdd(&ct, 1);
}



__device__ void syncBlocks(unsigned long long int* blockCounter){

    // const unsigned long long int fullMask = (1ULL << BLK_NUMS) - 1;
    // if (THID == 0)
    //  {
    //     while ((atomicOr(blockCounter, 1ULL << blockIdx.x)) != fullMask) { 
    //         // Busy wait... 
    //     }
    // }
    // __syncthreads();

    if (THID==0){
            atomicAdd(blockCounter, 1);
            printf("%d ", blockIdx.x);
            // __threadfence();
            while(ldg((unsigned int*)blockCounter)<BLK_NUMS){};
            // busy wait until all blocks increment
    }  
    __syncthreads();

}

__global__ void PKC(G_pointers d_p, unsigned int *global_count, int level, int V, 
                    unsigned long long int* blockCounter, unsigned int* glBuffers){

    // atomicAnd(&ct, 0);
    __shared__ unsigned int shBuffer[MAX_NV];
    __shared__ unsigned int bufTail;
    __shared__ unsigned int* glBuffer;
    __shared__ unsigned int base;
    __shared__ unsigned int lock;
    unsigned int warp_id = THID / 32;
    unsigned int lane_id = THID % 32;
    unsigned int regTail, regBase;
    unsigned int i;
    if(THID==0){
        bufTail = 0;
        base = 0;
        lock = 0;
        glBuffer = glBuffers + blockIdx.x*GLBUFFER_SIZE; 
        assert(glBuffer!=NULL);
    }

    
    __syncthreads();

    selectNodesAtLevel(d_p.degrees, V, shBuffer, glBuffer, &bufTail, level);

    syncBlocks(blockCounter);



    // bufTail is being incrmented within the loop, 
    // warps should process all the nodes added during the execution of loop
    
    // for(unsigned int i = warp_id; i<bufTail ; i +=warps_each_block ){
    // this for loop is a wrong choice, as many threads will exit from the loop checking the condition
    while(true){
        __syncthreads(); //syncthreads must be executed by all the threads
        regBase = base;
        regTail = bufTail;
        __syncthreads();

        if(THID == 0){
            // update base for next iteration
            base += WARPS_EACH_BLK;
            if(regTail < base )
                base = regTail;
        }

        if(regBase == regTail) break; // all the threads will evaluate to true at same iteration
        i = regBase + warp_id;
        if(i >= regTail) continue; // this warp won't have to do anything     
        
        
        unsigned int v, start, end;

        v = readFromBuffer(shBuffer, glBuffer, i);
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
                    writeToBuffer(shBuffer, glBuffer, loc, u);
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
        // if(glBuffer!=NULL) free(glBuffer);
    }
    // atomicAnd(&ct, 0);
}
