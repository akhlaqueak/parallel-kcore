
#include "../inc/device_funcs.h"
#include "stdio.h"

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
__device__ void selectNodesAtLevel(unsigned int *degrees, unsigned int V, unsigned int* buffer, unsigned int** helper, unsigned int* e, unsigned int level){

    unsigned int global_threadIdx = blockIdx.x * BLK_DIM + THID; 
    __shared__ bool predicate[BLK_DIM];
    __shared__ unsigned int addresses[BLK_DIM];
    
    for(unsigned int base = 0; base < V; base += N_THREADS){
        
        unsigned int v = base + global_threadIdx; 

        // all threads should get some value, if vertices are less than n_threads, rest of the threads get zero
        predicate[THID] = (v<V)? (degrees[v] == level) : 0;

        addresses[THID] = predicate[THID];

        exclusiveScan(addresses);
        
        //check if we need to allocate a helper for this block
        if(     
                (THID == BLK_DIM-1) && // only last thread in a block does this job
                // e[0]: no. of nodes already selected, addresses[...]: no. of nodes in currect scan
                (e[0] + addresses[THID] >= MAX_NV) &&  
                // check if it's not already allocated
                (helper[0] == NULL)
            ){
                // printf("Memory allocate in compact ");  
                helper[0] = (unsigned int*) malloc(sizeof(unsigned int) * HELPER_SIZE);            
                assert(helper[0]!=NULL);
        }
        
        // this sync is necessary so that memory is allocated before writing to buffer
        __syncthreads();
        
        if(predicate[THID]){
            unsigned int loc = addresses[THID] + e[0];
            writeToBuffer(buffer, helper, loc, v);
        }
        
        // this sync is necessary so that e[0] is updated after all threads have been written to buffer
        __syncthreads();
            
            
        if(THID == BLK_DIM - 1){            
            e[0] += (addresses[THID] + predicate[THID]);
        }
        
        __syncthreads();
            
    }
}

//todo: use inline and redue getwriteloc only to get loc, don't need helper
__device__ inline unsigned int getWriteLoc(unsigned int* e){
    unsigned int loc = atomicAdd(e, 1);
    return loc;
}

__device__ void writeToBuffer(unsigned int* buffer,  unsigned int** helper, unsigned int loc, unsigned int v){
    assert(loc < HELPER_SIZE + MAX_NV);
    if(loc < MAX_NV){
        buffer[loc] = v;
    }
    else{
        if(loc == MAX_NV){ // checking equal so that only one thread in a warp should allocate helper
            helper[0] = (unsigned int*) malloc(sizeof(unsigned int) * HELPER_SIZE); 
            printf("A ");
            assert(helper[0] != NULL); 
        }
        else while(helper[0]==NULL);
        
        helper[0][loc-MAX_NV] = v; 
    }
}


__device__ unsigned int readFromBuffer(unsigned int* buffer, unsigned int** helper, unsigned int loc){
    assert(loc < MAX_NV + HELPER_SIZE);
    return ( loc < MAX_NV ) ? buffer[loc] : helper[0][loc-MAX_NV]; 
}

__global__ void PKC(G_pointers d_p, unsigned int *global_count, int level, int V){
    
    
    __shared__ unsigned int buffer[MAX_NV];
    __shared__ unsigned int e;
    __shared__ unsigned int* helper;
    __shared__ unsigned int base;
    unsigned int warp_id = THID / 32;
    unsigned int lane_id = THID % 32;
    unsigned int i;

    if(THID == 0){
        e = 0;
        helper = NULL;
        base = 0;
    }

    __syncthreads();

    selectNodesAtLevel(d_p.degrees, V, buffer, &helper, &e, level);

    if(level == 1 && THID == 0) printf("%d ", e);
    
    // e is being incremented within the loop, 
    // warps should process all the nodes added during the execution of loop
    // for that purpose base is introduced, is incremented whenever a warp takes a job.
    // todo: busy waiting on several blocks

    // e = 10;
    // for(unsigned int i = warp_id; i<e ; i += WARPS_EACH_BLK){
    // this for loop is a wrong choice, as many threads will exit from the loop checking the condition     
    while(true){
        __syncthreads(); //syncthreads must be executed by all the threads, so can't put after break or continue...

        if(base == e) break;

        i = base + warp_id;
        
        if(THID == 0){
            base += WARPS_EACH_BLK;
            if(e < base )
                base = e;
        }
        __syncthreads();
        if(i >= e) continue; // this warp won't have to do anything 


        // only first lane reads buffer, start and end
        // it is then broadcasted to all lanes in the warp
        // it's done to reduce multiple accesses to global memory... 
        // todo: better if to read from global memory by all lanes

        
        unsigned int v = readFromBuffer(buffer, &helper, i);
        unsigned int start = d_p.neighbors_offset[v];
        unsigned int end = d_p.neighbors_offset[v+1];
        
        for(int j = start + lane_id; j<end ; j+=32){
            unsigned int u = d_p.neighbors[j];
            if(d_p.degrees[u] > level){
                unsigned int a = atomicSub(d_p.degrees+u, 1);
            
                if(a == level+1){
                    unsigned int loc = getWriteLoc(&e);
                    writeToBuffer(buffer, &helper, loc, u);
                    // printf("%d ", 1);
                }

                if(a <= level){
                    // node degree became less than the level after decrementing... 
                    atomicAdd(d_p.degrees+u, 1);
                }
            }
        }        
    }
    
    __syncthreads();

    if(THID == 0 && e!=0){
        atomicAdd(global_count, e); // atomic since contention among blocks
        if(helper!=NULL) free(helper);
    }

}


