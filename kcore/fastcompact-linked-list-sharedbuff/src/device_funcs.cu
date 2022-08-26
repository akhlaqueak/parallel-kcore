
#include "../inc/device_funcs.h"
// including these cuda files as inc files, so that to create a single compilation unit

#include "./buffer.cc"
#include "./scans.cc"
__device__ void selectNodesAtLevel(bool* predicate, volatile unsigned int* addresses, unsigned int* temp,
    unsigned int *degrees, unsigned int V, unsigned int* shBuffer, Node** tail, Node** head, unsigned int* bufTail, unsigned int level){


    unsigned int glThreadIdx = blockIdx.x * BLK_DIM + THID; 

    for(unsigned int base = 0; base < V; base += N_THREADS){
        
        unsigned int v = base + glThreadIdx; 

        // all threads should get some value, if vertices are less than n_threads, rest of the threads get zero
        predicate[THID] = (v<V)? (degrees[v] == level) : 0;
        temp[THID] = v;

        compactBlock(predicate, addresses, temp, shBuffer, tail, head, bufTail);        
        __syncthreads();
            
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
        };// busy wait until all blocks increment
    }   
    __syncthreads();
}


__global__ void PKC(G_pointers d_p, unsigned int *global_count, int level, int V, volatile unsigned int* blockCounter){
    
    
    __shared__ Node* tail;
    __shared__ Node* head;
    __shared__ unsigned int bufTail;
    __shared__ unsigned int base;
    __shared__ bool predicate[BLK_DIM];
    __shared__ unsigned int temp[BLK_DIM];
    __shared__ volatile unsigned int addresses[BLK_DIM];
    __shared__ unsigned int shBuffer[MAX_NV];
    __shared__ volatile unsigned int lock;

    unsigned int warp_id = THID / 32;
    unsigned int lane_id = THID % 32;
    unsigned int i;
    if(THID==0){
        tail = NULL;
        head = NULL;
        bufTail = 0;
        base = 0;
        lock = 0;
    }
    predicate[THID] = 0;
    
    selectNodesAtLevel(predicate, addresses, temp, d_p.degrees, V, shBuffer, &tail, &head, &bufTail, level);
    if(level == 1 && THID == 0) printf("%d ", bufTail);

    __syncthreads();

    // bufTail is being incremented within the loop, 
    // warps should process all the nodes added during the execution of loop
    // for that purpose base is introduced, is incremented whenever a warp takes a job.
    
    // done: busy waiting on several blocks

    syncBlocks(blockCounter);
    // bufTail = 10;
    // for(unsigned int i = warp_id; i<bufTail ; i += WARPS_EACH_BLK){
    // this for loop is a wrong choice, as many threads might exit from the loop checking the condition     
    while(true){
        __syncthreads(); //syncthreads must be executed by all the threads, so can't put after break or continue...
        if(base == bufTail) break;
        i = base + warp_id;
        
        if(THID==0 && head!=NULL)
            if(base >= head->limit){
                advanceNode(&head);
            }

        if(THID == 0){
            base += WARPS_EACH_BLK;
            if(bufTail < base )
                base = bufTail;
        }
        __syncthreads();
        if(i >= bufTail) continue; // this warp won't have to do anything 

        
        unsigned int v = readFromBuffer(shBuffer, head, i);
        unsigned int start = d_p.neighbors_offset[v];
        unsigned int end = d_p.neighbors_offset[v+1];
        // for(int j = start + lane_id; j<end ; j+=32){
        // the for loop may leave some of the threads inactive in its last iteration
        // following while loop will keep all threads active until the continue condition
        while(true){
            __syncwarp();

            compactWarp(predicate, addresses, temp, 
                        shBuffer, &tail, &head, &bufTail, &lock);
            __syncwarp();

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
                    // unsigned int loc = getWriteLoc(&bufTail);
                    // writeToBuffer(shBuffer, &glBuffer, loc, u);
                }

                if(a <= level){
                    // node degree became less than the level after decrementing... 
                    atomicAdd(d_p.degrees+u, 1);
                }
                // __threadfence();
            }
        }        
    }
    
    __syncthreads();

    if(THID == 0 && bufTail!=0){
        atomicAdd(global_count, bufTail); // atomic since contention among blocks
        if(head!=NULL) free(head);
    }

}


