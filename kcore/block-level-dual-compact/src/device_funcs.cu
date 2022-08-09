
#include "../inc/device_funcs.h"
#include "../inc/scans.h"
#include "stdio.h"


// __device__ inline unsigned int getWriteLoc(unsigned int* bufTail){
//     return atomicAdd(bufTail, 1);
// }

__device__ void writeToBuffer(unsigned int* shBuffer,    unsigned int* glBuffer, unsigned int loc, unsigned int v){
    assert(loc < GLBUFFER_SIZE + MAX_NV);
    if(loc < MAX_NV)
        shBuffer[loc] = v;
    else
        glBuffer[loc-MAX_NV] = v;
}


__device__ unsigned int readFromBuffer(unsigned int* shBuffer,   unsigned int* glBuffer, unsigned int loc){
    assert(loc < GLBUFFER_SIZE + MAX_NV);
    return ( loc < MAX_NV ) ? shBuffer[loc] : glBuffer[loc-MAX_NV]; 
}



__device__ inline bool allocationRequired( unsigned int* glBuffer, unsigned int loc, unsigned int dim){
    return (THID%dim == dim-1 && // last thread of warp or block
        glBuffer == NULL && // global buffer is not allocated before
        loc >= MAX_NV
    );
}
__device__ inline void allocateMemory( unsigned int** glBufferPtr){
        glBufferPtr[0] = ( unsigned int*) malloc(sizeof(unsigned int) * GLBUFFER_SIZE);
        // printf("allocated %d %d\n", blockIdx.x, THID);
        assert(glBufferPtr[0]!=NULL);        
}

__device__ void allocateMemoryMutex( unsigned int** glBufferPtr, unsigned int loc, volatile unsigned int* lock){
    if(atomicExch((unsigned int*)lock, 1) == 0){        
        // printf("mutex %d %d\n", blockIdx.x, THID);
        allocateMemory(glBufferPtr);
        lock[0] = 2; // not necessary to do it atomically, since it's the only thread in critical section
        __threadfence_block(); // it ensures the writes done by this thread are visible by all other threads in the block
    }
    while(lock[0]!=2);
}    

__device__ void syncBlocks(volatile unsigned int* blockCounter){
    
    if (THID==0)
    {
        atomicAdd((unsigned int*)blockCounter, 1);
        __threadfence();
        while(blockCounter[0]<BLK_NUMS){
            // number of blocks can't be greater than SMs, else it'll cause infinite loop... 
            // printf("%d ", blockCounter[0]);
        };// busy wait until all blocks increment
    }
    

    
    __syncthreads();
}

__global__ void PKC(G_pointers d_p, unsigned int *global_count, int level, int V, volatile unsigned int* blockCounter){
    
    
    __shared__ unsigned int shBuffer[MAX_NV];
    __shared__ unsigned int bufTail;
    __shared__  unsigned int* glBuffer;
    __shared__ unsigned int base;
    __shared__ unsigned int predicate[BLK_DIM];
    __shared__ unsigned int temp[BLK_DIM];
    __shared__ unsigned int addresses[BLK_DIM];
    __shared__ volatile unsigned int allocLock;
    __shared__ volatile unsigned int readLock;

    unsigned int warp_id = THID / 32;
    unsigned int lane_id = THID % 32;
    unsigned int i;

    bufTail = 0;
    glBuffer = NULL;
    base = 0;
    predicate[THID] = 0;
    allocLock = 0;
    readLock = 0;

    compactBlock(d_p.degrees, V, shBuffer, &glBuffer, &bufTail, level);
    // if(level == 1 && THID == 0) printf("%d ", bufTail);

    __syncthreads();

    
    // bufTail is being incremented within the loop, 
    // warps should process all the nodes added during the execution of loop
    // for that purpose base is introduced, is incremented whenever a warp takes a job.
    
    // todo: busy waiting on several blocks

    syncBlocks(blockCounter);
    // bufTail = 10;
    // for(unsigned int i = warp_id; i<bufTail ; i += WARPS_EACH_BLK){
    // this for loop is a wrong choice, as many threads might exit from the loop checking the condition     
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

        
        unsigned int v = readFromBuffer(shBuffer, glBuffer, i);
        unsigned int start = d_p.neighbors_offset[v];
        unsigned int end = d_p.neighbors_offset[v+1];
        unsigned int b1 = start;
        // for(int j = start + lane_id; j<end ; j+=32){
        // the for loop may leave some of the threads inactive in its last iteration
        // following while loop will keep all threads active until the continue condition
        while(true){
            __syncwarp();

            compactWarp(temp+(warp_id*WARP_SIZE), addresses+(warp_id*WARP_SIZE), predicate+(warp_id*WARP_SIZE), shBuffer, &glBuffer, &bufTail, &allocLock);
            
            if(b1 >= end) break;

            unsigned int j = b1 + lane_id;
            b1 += WARP_SIZE;
            if(j >= end) continue;

            unsigned int u = d_p.neighbors[j];
            if(d_p.degrees[u] > level){
                unsigned int a = atomicSub(d_p.degrees+u, 1);
            
                if(a == level+1){
                    temp[THID] = u;
                    predicate[THID] = 1;
                    // unsigned int loc = getWriteLoc(&bufTail);
                    // writeToBuffer(shBuffer, &glBuffer, loc, u);
                }

                if(a <= level){
                    // node degree became less than the level after decrementing... 
                    atomicAdd(d_p.degrees+u, 1);
                }
                __threadfence();
            }
        }        
    }
    
    __syncthreads();

    if(THID == 0 && bufTail!=0){
        atomicAdd(global_count, bufTail); // atomic since contention among blocks
        if(glBuffer!=NULL) free((unsigned int*)glBuffer);
    }

}


