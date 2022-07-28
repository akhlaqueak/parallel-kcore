
#include "../inc/device_funcs.h"
#include "stdio.h"
__device__ void writeToBuffer(unsigned int* buffer,  unsigned int** helper, unsigned int* e, unsigned int v){
    unsigned int loc = atomicAdd(e, 1);
    assert(e[0] < HELPER_SIZE + MAX_NV);

    if(loc == MAX_NV){ // checking equal so that only one thread in a warp should allocate helper
        helper[0] = (unsigned int*) malloc(HELPER_SIZE); 
        assert(helper[0] != NULL); 
    }
    __syncwarp();
    
    if(loc < MAX_NV){
        buffer[loc] = v;
    }
    else{
        helper[0][loc-MAX_NV] = v; 
    }
}

__device__ void selectNodesAtLevel(unsigned int *degrees, unsigned int V, unsigned int* buffer, unsigned int** helper, unsigned int* e, unsigned int level){
    unsigned int global_threadIdx = blockIdx.x * blockDim.x + threadIdx.x; 
    for(unsigned int i=global_threadIdx; i< V; i+= N_THREADS){
        if(degrees[i] == level){
            writeToBuffer(buffer, helper, e , i);
        }
    }
}

__device__ unsigned int readFromBuffer(unsigned int* buffer, unsigned int** helper, unsigned int loc){
    assert(loc < MAX_NV + HELPER_SIZE);
    return ( loc < MAX_NV ) ? buffer[loc] : helper[0][loc-MAX_NV]; 
}

__global__ void PKC(G_pointers d_p, unsigned int *global_count, int level, int V){


    __shared__ unsigned int buffer[WARPS_EACH_BLK*MAX_NV];
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

	
    // DONE: remove the warp level implementations, go to block level.
    __syncwarp();

    selectNodesAtLevel(d_p.degrees, V, buffer, &helper, &e, level);


    // TODO: Need to look into the issue when e < WARPS_EACH_BLK

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
            printf("%d-%d-%d ", e , e_processed, i);
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
        }

        __syncthreads();
    }

    if(THID == 0 && e!=0){
        atomicAdd(global_count, e); // atomic since contention among blocks
        if(helper!=NULL) free(helper);
    }

}


