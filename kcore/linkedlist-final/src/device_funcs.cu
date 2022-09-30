
#include "../inc/device_funcs.h"
// including these cuda files as inc files, so that to create a single compilation unit

#include "./buffer.cc"
#include "./scans.cc"
__global__ void selectNodesAtLevel(unsigned int* degrees, unsigned int level, 
                        unsigned int V, unsigned int* bufTails,
                        Node** heads, Node** tails){
    __shared__ unsigned int bufTail;
    __shared__ unsigned int temp[BLK_DIM]; 
    __shared__ volatile unsigned int addresses[BLK_DIM];
    __shared__ bool predicate[BLK_DIM];
    __shared__ Node** head;
    __shared__ Node** tail;
    __shared__ volatile unsigned int lock;

    if(THID == 0){
        head = heads + blockIdx.x;
        tail = tails + blockIdx.x;
        bufTail = bufTails[blockIdx.x];
        lock = 0;
        // head[0] = NULL;
        // tail[0] = NULL;
        // bufTail = 0;
    }
    predicate[THID] = 0;
    __syncthreads();
    
    unsigned int glThreadIdx = blockIdx.x * BLK_DIM + THID; 

    for(unsigned int base = 0; base < V; base += N_THREADS){
        
        unsigned int v = base + glThreadIdx; 

        // all threads should get some value, if vertices are less than n_threads, rest of the threads get zero
        predicate[THID] = (v<V)? (degrees[v] == level) : 0;
        temp[THID] = v;
        compactWarp(predicate, addresses, temp, tail, head, &bufTail, &lock);        
        __syncthreads();
            
    }

    if(THID == 0){
        heads[blockIdx.x] = head[0];
        tails[blockIdx.x] = tail[0];
        bufTails[blockIdx.x] = bufTail;
    }
}


__global__ void processNodes(G_pointers d_p, int level, int V, unsigned int* bufTails, 
    unsigned int* global_count, Node** heads, Node** tails){

    
    __shared__ Node** tail;
    __shared__ Node** head;
    __shared__ unsigned int bufTail;
    __shared__ unsigned int base;
    __shared__ bool predicate[BLK_DIM];
    __shared__ unsigned int temp[BLK_DIM];
    __shared__ volatile unsigned int addresses[BLK_DIM];
    __shared__ volatile unsigned int lock;

    unsigned int warp_id = THID / 32;
    unsigned int lane_id = THID % 32;
    unsigned int i, regbase, regtail;
    if(THID==0){
        tail = tails + blockIdx.x;
        head = heads + blockIdx.x;
        bufTail = bufTails[blockIdx.x];
        base = 0;
        lock = 0;
    }
    predicate[THID] = 0;
    

    // bufTail is being incremented within the loop, 
    // warps should process all the nodes added during the execution of loop
    // for that purpose base is introduced, is incremented whenever a warp takes a job.
    

    // bufTail = 10;
    // for(unsigned int i = warp_id; i<bufTail ; i += WARPS_EACH_BLK){
    // this for loop is a wrong choice, as many threads might exit from the loop checking the condition     
    while(true){
        __syncthreads(); //syncthreads must be executed by all the threads, so can't put after break or continue...
        if(base == bufTail) break;
        i = base + warp_id;
        regbase = base;
        regtail = bufTail;
        __syncthreads();
        
        if(THID==0 && head[0]!=NULL){
                // printf("%d.%d.", regbase, head[0]->limit);
            if(regbase >= head[0]->limit){
                deleteHead(head);
            }
        }
        __syncthreads();

        if(THID == 0){
            base += WARPS_EACH_BLK;
            if(regtail < base )
                base = regtail;
        }
        __syncthreads();
        if(i >= regtail) continue; // this warp won't have to do anything 

        
        unsigned int v = readFromBuffer(head[0], i);
        unsigned int start = d_p.neighbors_offset[v];
        unsigned int end = d_p.neighbors_offset[v+1];
        // for(int j = start + lane_id; j<end ; j+=32){
        // the for loop may leave some of the threads inactive in its last iteration
        // following while loop will keep all threads active until the continue condition
        while(true){
            __syncwarp();
            compactWarp(predicate, addresses, temp, 
                        tail, head, &bufTail, &lock);
            predicate[THID] = 0;
            __syncwarp();

            if(start >= end) break;

            unsigned int j = start + lane_id;
            start += WARP_SIZE;
            if(j >= end) continue;

            unsigned int u = d_p.neighbors[j];
            if(*(d_p.degrees+u) > level){
                unsigned int a = atomicSub(d_p.degrees+u, 1);
                
                if(a == level+1){
                    temp[THID] = u;
                    predicate[THID] = 1;
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
        if(head[0]!=NULL) {
            printf("/");
            free(head[0]);
        }
    }

}