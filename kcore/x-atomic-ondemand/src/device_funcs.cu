
#include "../inc/device_funcs.h"
#include "stdio.h"
#include "buffer.cc"


__device__ void selectNodesAtLevel(unsigned int *degrees, unsigned int V, unsigned int* shBuffer, unsigned int** glBuffer, unsigned int* bufTail, unsigned int level){
    unsigned int global_threadIdx = blockIdx.x * blockDim.x + threadIdx.x; 
    for(unsigned int base = 0; base < V; base += N_THREADS){
        
        unsigned int v = base + global_threadIdx; 

        // all threads should get some value, if vertices are less than n_threads, rest of the threads get zero

        // only the last thread in warp is responsible to alloate memory
        // adding blk_dim to anticipate allocation
        if(THID==BLK_DIM-1){
            if(allocationRequired(glBuffer[0], bufTail[0]+BLK_DIM)){
                allocateMemory(glBuffer);
            }
        }

        __syncthreads();

        if(v >= V) continue;

        if(degrees[v] == level){
            unsigned int loc = atomicAdd(bufTail, 1);
            writeToBuffer(shBuffer, glBuffer[0], loc, v);
        }
    }
}



__device__ void syncBlocks(volatile unsigned int* blockCounter){
    
    if (THID==0)
    {
        atomicAdd((unsigned int*)blockCounter, 1);
        __threadfence();
        while(blockCounter[0]<BLK_NUMS){
            // number of blocks can't be greater than SMs, else it'll cause infinite loop... 
            // printf("%d ", blockCounter[0]);
        }// busy wait until all blocks increment
    }   
    __syncthreads();
}

__global__ void PKC(G_pointers d_p, unsigned int *global_count, int level, int V, volatile unsigned int* blockCounter){


    __shared__ unsigned int shBuffer[MAX_NV];
    __shared__ unsigned int bufTail;
    __shared__ unsigned int* glBuffer;
    __shared__ unsigned int base;
    __shared__ unsigned int lock;
    unsigned int warp_id = THID / 32;
    unsigned int lane_id = THID % 32;
    unsigned int i;

    // if(THID == 0){
        bufTail = 0;
        glBuffer = NULL;
        base = 0;
        lock = 0;
        // glBuffer = (unsigned int*) malloc(sizeof(unsigned int) * GLBUFFER_SIZE); 
    // }

    __syncthreads();

    selectNodesAtLevel(d_p.degrees, V, shBuffer, &glBuffer, &bufTail, level);

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

        v = readFromBuffer(shBuffer, glBuffer, i);
        start = d_p.neighbors_offset[v];
        end = d_p.neighbors_offset[v+1];


        while(true){
            __syncwarp();

            if(start >= end) break;

            unsigned int j = start + lane_id;
            start += WARP_SIZE;
            if(j >= end) continue;
            if(lane_id==WARP_SIZE-1 && 
                allocationRequired(glBuffer, bufTail+WARP_SIZE)){
                allocateMemoryMutex(&glBuffer, &lock);
            }

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
        if(glBuffer!=NULL) free(glBuffer);
    }

}

