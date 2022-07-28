
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
    
    for(unsigned int i = 0; i < V; i+= N_THREADS){
        
        unsigned int v = i + global_threadIdx; 

        // all threads should get some value, if vertices are less than n_threads, rest of the threads get zero
        predicate[THID] = (v<V)? (degrees[v] == level) : 0;

        addresses[THID] = predicate[THID];

        exclusiveScan(addresses);

        __syncthreads();

        
        

        if(     //check if we need to allocate a helper for this block
            (THID == BLK_DIM-1) && // only one thread in a block does this job
                // e[0]: no. of nodes already selected, addresses[...]: no. of nodes in currect scan
            (e[0] + addresses[THID] >= MAX_NV) &&  
                // check if it's not already allocated
                (helper[0] == NULL)
            ){
                
                helper[0] = (unsigned int*) malloc(HELPER_SIZE);            
                assert(helper[0]!=NULL);
        }
        __syncthreads();
        
        if(predicate[THID]){
            unsigned int loc = addresses[THID] + e[0];
            if(loc < MAX_NV)
                buffer[loc] = v;
                else
                helper[0][loc - MAX_NV]  = v;   
            }
            
            
            __syncthreads();
            
            if(THID == BLK_DIM - 1){
                e[0] += addresses[THID];
                printf("%d*%d ", blockIdx.x, e[0]);
            }
            
            __syncthreads();
            
        }
}


__device__ void writeToBuffer(unsigned int* buffer,  unsigned int** helper, unsigned int* e, unsigned int v){
    unsigned int loc = atomicAdd(e, 1);
    assert(e[0] < HELPER_SIZE + MAX_NV);

    if(loc == MAX_NV){ // checking equal so that only one thread in a warp should allocate helper
        helper[0] = (unsigned int*) malloc(HELPER_SIZE); 
        assert(helper[0] != NULL); 
    }
    __syncthreads();
    
    if(loc < MAX_NV){
        buffer[loc] = v;
    }
    else{
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
    __shared__ unsigned int e_processed;

    if(THID == 0){
        e = 0;
        helper = NULL;
        e_processed = 0;
    }

    unsigned int warp_id = THID / 32;
    unsigned int lane_id = THID % 32;

	
    __syncthreads();

    selectNodesAtLevel(d_p.degrees, V, buffer, &helper, &e, level);

    __syncthreads();

    // e is being incrmented within the loop, 
    // warps should process all the nodes added during the execution of loop
    // for that purpose e_processes is introduced, is incremented whenever a warp takes a job. 
    for(unsigned int i = warp_id; i<e ; i = warp_id + e_processed){
    
        unsigned int v, start, end;

        // only first lane reads buffer, start and end
        // it is then broadcasted to all lanes in the warp
        // it's done to reduce multiple accesses to global memory... 

        if(lane_id == 0){ 
            v = readFromBuffer(buffer, &helper, i);
            start = d_p.neighbors_offset[v];
            end = d_p.neighbors_offset[v+1];
            atomicAdd(&e_processed, 1);
        }

        v = __shfl_sync(0xFFFFFFFF, v, 0);
        start = __shfl_sync(0xFFFFFFFF, start, 0);
        end = __shfl_sync(0xFFFFFFFF, end, 0);

        for(int j = start + lane_id; j<end ; j+=32){
            unsigned int u = d_p.neighbors[j];
            if(d_p.degrees[u] > level){
                unsigned int a = atomicSub(d_p.degrees+u, 1);
            
                if(a == level+1){
                    writeToBuffer(buffer, &helper, &e, u);
                }

                if(a <= level){
                    // node degree became less than the level after decrementing... 
                    atomicAdd(d_p.degrees+u, 1);
                }
            }
            __syncwarp();
        }

        __syncthreads();
    }
    __syncthreads();

    if(THID == 0 && e!=0){
        atomicAdd(global_count, e); // atomic since contention among blocks
        if(helper!=NULL) free(helper);
    }

}


