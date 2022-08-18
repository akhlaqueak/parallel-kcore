
#include "../inc/device_funcs.h"
#include "stdio.h"

__device__ unsigned long long int blockCounter;

__device__ void syncBlocks(){
    
    
    const auto SollMask = (1 << gridDim.x) - 1;
    if (THID == 0) {
        while ((atomicOr( blockCounter, 1ULL << blockIdx.x)) != SollMask) { /*do nothing*/ }
    }
    // if (ThreadId() == 0 && 0 == blockIdx.x) {
    //     printf("Print a single line for the entire process")
    // }
    
}
__device__ void exclusiveScan(unsigned int* addresses){

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




__device__ void selectNodesAtLevel(unsigned int *degrees, unsigned int V, unsigned int* shBuffer,  unsigned int** glBuffer, unsigned int* bufTail, unsigned int level, unsigned int* lock){

    unsigned int global_threadIdx = blockIdx.x * BLK_DIM + THID; 
    __shared__ bool predicate[BLK_DIM];
    __shared__ unsigned int addresses[BLK_DIM];
    
    for(unsigned int base = 0; base < V; base += N_THREADS){
        
        unsigned int v = base + global_threadIdx; 

        // all threads should get some value, if vertices are less than n_threads, rest of the threads get zero
        predicate[THID] = (v<V)? (degrees[v] == level) : 0;

        addresses[THID] = predicate[THID];

        exclusiveScan(addresses);
        
        //check if we need to allocate a glBuffer for this block
        // if(     
        //         (THID == BLK_DIM-1) && // only last thread in a block does this job
        //         // bufTail[0]: no. of nodes already selected, addresses[...]: no. of nodes in currect scan
        //         (bufTail[0] + addresses[THID] >= MAX_NV) &&  
        //         // check if it's not already allocated
        //         (glBuffer[0] == NULL)
        //     ){
        //         // printf("Memory allocate in compact ");  
        //         glBuffer[0] = (unsigned int*) malloc(sizeof(unsigned int) * GLBUFFER_SIZE);            
        //         assert(glBuffer[0]!=NULL);
        // }
        
        // this sync is necessary so that memory is allocated before writing to shBuffer
        __syncthreads();
        
        if(predicate[THID]){
            unsigned int loc = addresses[THID] + bufTail[0];
            writeToBuffer(shBuffer, glBuffer, loc, v, lock);
        }
        
        // this sync is necessary so that bufTail[0] is updated after all threads have been written to shBuffer
        __syncthreads();
            
            
        if(THID == BLK_DIM - 1){            
            bufTail[0] += (addresses[THID] + predicate[THID]);
        }
        
        __syncthreads();
            
    }
}

//todo: use inline and redue getwriteloc only to get loc, don't need glBuffer
__device__ inline unsigned int getWriteLoc(unsigned int* bufTail){
    return atomicAdd(bufTail, 1);
}

__device__ void writeToBuffer(unsigned int* shBuffer,   unsigned int** glBuffer_p, unsigned int loc, unsigned int v, unsigned int* lock){
    assert(loc < GLBUFFER_SIZE + MAX_NV);
    if(loc < MAX_NV){
        shBuffer[loc] = v;
    }
    else{
        // if(loc == MAX_NV){ // checking equal so that only one thread in a warp should allocate glBuffer
        //     glBuffer_p[0] = ( unsigned int*) malloc(sizeof(unsigned int) * GLBUFFER_SIZE); 
        //     assert(glBuffer_p[0] != NULL); 
        // }
        // else while(glBuffer_p[0]==NULL)
        // //  printf("1");
        //  ; // busy wait until glBuffer is allocated 
        
        // while(lock[0]!=2)
        //     if(atomicExch(lock, 1) == 0){
        //         glBuffer_p[0] = ( unsigned int*) malloc(sizeof(unsigned int) * GLBUFFER_SIZE); 
        //         atomicExch(lock, 2);
        //     }
        assert(glBuffer_p[0]!=NULL);
        glBuffer_p[0][loc-MAX_NV] = v; 
    }
}


__device__ unsigned int readFromBuffer(unsigned int* shBuffer,  unsigned int* glBuffer, unsigned int loc){
    assert(loc < MAX_NV + GLBUFFER_SIZE);
    return ( loc < MAX_NV ) ? shBuffer[loc] : glBuffer[loc-MAX_NV]; 
}

__global__ void PKC(G_pointers d_p, unsigned int *global_count, int level, int V){
    
    
    __shared__ unsigned int shBuffer[MAX_NV];
    __shared__ unsigned int bufTail;
    __shared__  unsigned int* glBuffer;
    __shared__ unsigned int base;
    unsigned int warp_id = THID / 32;
    unsigned int lane_id = THID % 32;
    __shared__ unsigned int lock;
    unsigned int i;

    if(THID == 0){
        bufTail = 0;
        glBuffer = NULL;
        base = 0;
        lock = 0;
        glBuffer = (unsigned int*)malloc(sizeof(unsigned int)*GLBUFFER_SIZE);
        atomicAnd(&blockCounter, 0);
    }

    __syncthreads();

    selectNodesAtLevel(d_p.degrees, V, shBuffer, &glBuffer, &bufTail, level, &lock);

    syncBlocks();
    // if(level == 1 && THID == 0) printf("%d ", bufTail);
    
    // bufTail is being incremented within the loop, 
    // warps should process all the nodes added during the execution of loop
    // for that purpose base is introduced, is incremented whenever a warp takes a job.
    // todo: busy waiting on several blocks

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


        // only first lane reads shBuffer, start and end
        // it is then broadcasted to all lanes in the warp
        // it's done to reduce multiple accesses to global memory... 
        // todo: better if to read from global memory by all lanes

        
        unsigned int v = readFromBuffer(shBuffer, glBuffer, i);
        unsigned int start = d_p.neighbors_offset[v];
        unsigned int end = d_p.neighbors_offset[v+1];
        // unsigned int b1 = start;
        // while(true){
        //     __syncwarp();
        //     if(b1 >= end) break;
        //     unsigned int j = b1 + lane_id;
        //     b1 += 32;
        //     if(j >= end) continue;
        
        for(int j = start + lane_id; j<end ; j+=32){

            unsigned int u = d_p.neighbors[j];
            if(d_p.degrees[u] > level){
                unsigned int a = atomicSub(d_p.degrees+u, 1);
            
                if(a == level+1){
                    unsigned int loc = getWriteLoc(&bufTail);
                    writeToBuffer(shBuffer, &glBuffer, loc, u, &lock);
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


