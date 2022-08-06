
#include "../inc/device_funcs.h"
#include "stdio.h"


__device__ void scanBlock(unsigned int* addresses){

    for (int d = 2; d <= BLK_DIM; d = d*2) {   
        __syncthreads();  
        if (THID % d == d-1)  
            addresses[THID] += addresses[THID-d/2];  
    }

    if(THID == (BLK_DIM-1)) {
        addresses[THID] = 0;
    }

    for(int d=BLK_DIM; d > 1; d/=2){
        __syncthreads();
        if(THID % d == d-1){
            unsigned int val = addresses[THID-d/2];
            addresses[THID-d/2] = addresses[THID];
            addresses[THID] += val;
        }
    }
}

__device__ void scanWarp(unsigned int* addresses){
    unsigned int lane_id = THID%WARP_SIZE;
    for (int d = 2; d <= WARP_SIZE; d = d*2) {   
        __syncwarp();  
        if (lane_id % d == d-1)  
            addresses[lane_id] += addresses[lane_id-d/2];  
    }

    if(lane_id == (WARP_SIZE-1)) {
        addresses[lane_id] = 0;
    }

    for(int d=WARP_SIZE; d > 1; d/=2){
        __syncwarp();
        if(lane_id % d == d-1){
            unsigned int val = addresses[lane_id-d/2];
            addresses[lane_id-d/2] = addresses[lane_id];
            addresses[lane_id] += val;
        }
    }
    __syncwarp();
}

__device__ void compactBlock(unsigned int *degrees, unsigned int V, unsigned int* shBuffer,   unsigned int** glBufferPtr, unsigned int* bufTailPtr, unsigned int level){

    unsigned int global_threadIdx = blockIdx.x * BLK_DIM + THID; 
    __shared__ bool predicate[BLK_DIM];
    __shared__ unsigned int addresses[BLK_DIM];
    
    for(unsigned int base = 0; base < V; base += N_THREADS){
        
        unsigned int v = base + global_threadIdx; 

        // all threads should get some value, if vertices are less than n_threads, rest of the threads get zero
        predicate[THID] = (v<V)? (degrees[v] == level) : 0;

        addresses[THID] = predicate[THID];

        scanBlock(addresses);
        
        addresses[THID] += bufTailPtr[0];


        if(allocationRequired(glBufferPtr[0], addresses[THID], BLK_DIM))
            allocateMemory(glBufferPtr);

        // this sync is necessary so that memory is allocated before writing to buffer
        __syncthreads();
        
        if(predicate[THID])
            writeToBuffer(shBuffer, glBufferPtr[0], addresses[THID], v);
        
        if(THID == BLK_DIM - 1){            
            bufTailPtr[0] += (addresses[THID] + predicate[THID]);
        }
        
        __syncthreads();
            
    }
}

__device__ void compactWarp(unsigned int* temp, unsigned int* addresses, unsigned int* predicate, 
    unsigned int* shBuffer,  unsigned int** glBufferPtr, unsigned int* bufTailPtr, volatile unsigned int* lock){
    
    unsigned int lane_id = THID%WARP_SIZE;

    unsigned int bTail;
    
    addresses[lane_id] = predicate[lane_id];

    scanWarp(addresses);
    // todo: look for atomic add at warp level.
    
    if(lane_id == WARP_SIZE-1){
        bTail = atomicAdd(bufTailPtr, addresses[lane_id]+predicate[lane_id]);
    }
    
    bTail = __shfl_sync(0xFFFFFFFF, bTail, WARP_SIZE-1);
    
    addresses[lane_id]+=bTail;


    if(allocationRequired(glBufferPtr[0], addresses[lane_id], WARP_SIZE)){
        // printf("trying allocation at: %d %d \n", blockIdx.x, THID);
        allocateMemoryMutex(glBufferPtr, addresses[lane_id], lock);    
    }
    __syncwarp();

    if(predicate[lane_id])
        writeToBuffer(shBuffer, glBufferPtr[0], addresses[lane_id], temp[lane_id]);

    predicate[lane_id] = 0;
}

// __device__ inline unsigned int getWriteLoc(unsigned int* bufTail){
//     return atomicAdd(bufTail, 1);
// }

__device__ void writeToBuffer(unsigned int* shBuffer,    unsigned int* glBuffer, unsigned int loc, unsigned int v){
    // if(blockIdx.x == 10) printf("loc: %d ", loc);
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
        printf("allocated %d %d\n", blockIdx.x, THID);
        assert(glBufferPtr[0]!=NULL);        
}

__device__ void allocateMemoryMutex( unsigned int** glBufferPtr, unsigned int loc, volatile unsigned int* lock){
    if(atomicExch((unsigned int*)lock, 1) == 0){        
        printf("mutex %d %d\n", blockIdx.x, THID);
        allocateMemory(glBufferPtr);
        lock[0] = 2; // not necessary to do it atomically, since it's the only thread in critical section
        __threadfence_block(); // it ensures the writes done by this thread are visible by all other threads in the block
    }
    while(lock[0]!=2);
}    

__device__ void synchronizeBlocks(volatile unsigned int* blockCounter){
    
    if (THID==0)
    {
        unsigned int val = atomicAdd((unsigned int*)blockCounter, 1);
        printf("%d ", val);
        __threadfence();
    }
    
    if(THID==0){
        while(blockCounter[0]<BLK_NUMS);// busy wait until all blocks increment
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
    __shared__ volatile unsigned int lock;

    unsigned int warp_id = THID / 32;
    unsigned int lane_id = THID % 32;
    unsigned int i;

    bufTail = 0;
    glBuffer = NULL;
    base = 0;
    predicate[THID] = 0;
    lock = 0;
    blockCounter[0] = 0;
    
    compactBlock(d_p.degrees, V, shBuffer, &glBuffer, &bufTail, level);
    // if(level == 1 && THID == 0) printf("%d ", bufTail);



    
    // bufTail is being incremented within the loop, 
    // warps should process all the nodes added during the execution of loop
    // for that purpose base is introduced, is incremented whenever a warp takes a job.
    
    // todo: busy waiting on several blocks

    synchronizeBlocks(blockCounter);
    // bufTail = 10;
    // for(unsigned int i = warp_id; i<bufTail ; i += WARPS_EACH_BLK){
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

        
        unsigned int v = readFromBuffer(shBuffer, glBuffer, i);
        unsigned int start = d_p.neighbors_offset[v];
        unsigned int end = d_p.neighbors_offset[v+1];
        unsigned int b1 = start;
        // for(int j = start + lane_id; j<end ; j+=32){
        // the for loop may leave some of the threads inactive in its last iteration
        // following while loop will keep all threads active until the continue condition
        while(true){
            __syncwarp();

            compactWarp(temp+(warp_id*WARP_SIZE), addresses+(warp_id*WARP_SIZE), predicate+(warp_id*WARP_SIZE), shBuffer, &glBuffer, &bufTail, &lock);
            
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
                }else{
                    predicate[THID] = 0;
                }

                if(a <= level){
                    // node degree became less than the level after decrementing... 
                    atomicAdd(d_p.degrees+u, 1);
                }
            }
        }        
    }
    
    __syncthreads();

    if(THID == 0 && bufTail!=0){
        atomicAdd(global_count, bufTail); // atomic since contention among blocks
        if(glBuffer!=NULL) free((unsigned int*)glBuffer);
    }

}


