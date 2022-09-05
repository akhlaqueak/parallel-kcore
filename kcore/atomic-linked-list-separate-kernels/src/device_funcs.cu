
#include "../inc/device_funcs.h"
// including these cuda files as inc files, so that to create a single compilation unit

#include "./buffer.cc"
#include "./scans.cc"



__global__ void initialScan(G_pointers d_p, unsigned int *global_count, int level, int V, unsigned int* bufTails, Node** tails, Node** heads){
    unsigned int global_threadIdx = blockIdx.x * blockDim.x + threadIdx.x; 
    Node** tail = tails + blockIdx.x;
    Node** head = heads + blockIdx.x;
    unsigned int* bufTail = bufTails + blockIdx.x;



    for(unsigned int base = 0; base < V; base += N_THREADS){
        
        unsigned int v = base + global_threadIdx; 

        // all threads should get some value, if vertices are less than n_threads, rest of the threads get zero

        // only the last thread in warp is responsible to alloate memory
        if(THID == BLK_DIM - 1){  

            if(allocationRequired(tail[0], bufTail[0]+BLK_DIM)) // adding BLK_DIM to bufTail so that node is created when possibly all threads need to add into it
                allocateMemory(tail, head);
        }
        __syncthreads();

        if(v >= V) continue;

        if(d_p.degrees[v] == level){
            unsigned int loc = atomicAdd(bufTail, 1);
            writeToBuffer(tail[0], loc, v);
        }
    }
}


__global__ void PKC(G_pointers d_p, unsigned int *global_count, int level, int V, unsigned int* bufTails, Node** tails, Node** heads){
   
    
    __shared__ Node** tail;
    __shared__ Node** head;
    __shared__ unsigned int bufTail;
    __shared__ unsigned int base;
    // __shared__ unsigned int predicate[BLK_DIM];
    // __shared__ unsigned int temp[BLK_DIM];
    // __shared__ unsigned int addresses[BLK_DIM];
    __shared__ volatile unsigned int lock;

    unsigned int warp_id = THID / 32;
    unsigned int lane_id = THID % 32;
    unsigned int i;

    if(THID==0){
        tail = tails + blockIdx.x;
        head = heads + blockIdx.x;
        bufTail = bufTails[blockIdx.x];
        base = 0;
        lock = 0;
    }

    // if(level == 1 && THID == 0) printf("%d ", bufTail);

    // __syncthreads();

    // bufTail is being incremented within the loop, 
    // warps should process all the nodes added during the execution of loop
    // for that purpose base is introduced, is incremented whenever a warp takes a job.
    
    // todo: busy waiting on several blocks

    // syncBlocks(blockCounter);
    // bufTail = 10;
    // for(unsigned int i = warp_id; i<bufTail ; i += WARPS_EACH_BLK){
    // this for loop is a wrong choice, as many threads might exit from the loop checking the condition     
    while(true){
        __syncthreads(); //syncthreads must be executed by all the threads, so can't put after break or continue...
        
        if(base == bufTail) break;
        
        i = base + warp_id;
        
        __syncthreads();
        
        if(THID == 0){
            assert(head!=NULL);
            if(base >= head[0]->limit){
                advanceNode(head);
            }
            base += WARPS_EACH_BLK;
            if(bufTail < base )
                base = bufTail;
        }
        __syncthreads();

        if(i >= bufTail) continue; // this warp won't have to do anything 

        
        unsigned int v = readFromBuffer(head[0], i);
        unsigned int start = d_p.neighbors_offset[v];
        unsigned int end = d_p.neighbors_offset[v+1];

        // for(int j = start + lane_id; j<end ; j+=32){
        // the for loop may leave some of the threads inactive in its last iteration
        // following while loop will keep all threads active until the continue condition
        while(true){


            if(start >= end) break;

            unsigned int j = start + lane_id;
            start += WARP_SIZE;

            // look for allocating the node:            
            if(THID == WARP_SIZE - 1){  
                if(allocationRequired(tail[0], bufTail+WARP_SIZE)){ // adding BLK_DIM to bufTail so that node is created when possibly all threads need to add into it
                    atomicCAS((unsigned int*)&lock, 2, 0); // resets the lock in case a memory was allocated before
                    __threadfence_block();
                    allocateMemoryMutex(tail, head, &lock);
                }
            }
            __syncwarp();

            if(j >= end) continue;

            unsigned int u = d_p.neighbors[j];
            if(ldg(d_p.degrees+u) > level){
                unsigned int a = atomicSub(d_p.degrees+u, 1);
                
                if(a == level+1){
                    // temp[THID] = u;
                    // predicate[THID] = 1;
                    unsigned int loc = atomicAdd(&bufTail, 1);
                    writeToBuffer(tail[0], loc, u);

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


