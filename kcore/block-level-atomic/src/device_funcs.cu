
#include "../inc/device_funcs.h"
#include "stdio.h"
__device__ unsigned int getWriteLoc(unsigned int** helper, unsigned int* e){
    unsigned int loc = atomicAdd(e, 1);
    assert(loc < HELPER_SIZE + MAX_NV);

    if(loc == MAX_NV){ // checking equal so that only one thread in a warp should allocate helper
        helper[0] = (unsigned int*) malloc(sizeof(unsigned int) * HELPER_SIZE); 
        assert(helper[0] != NULL); 
    }
    return loc;
}

__device__ void writeToBuffer(unsigned int* buffer,  unsigned int** helper, unsigned int loc, unsigned int v){
    assert(loc < HELPER_SIZE + MAX_NV);
    if(loc < MAX_NV){
        buffer[loc] = v;
    }
    else{
        assert(helper[0]!=NULL);
        helper[0][loc-MAX_NV] = v; 
    }
}

__device__ void selectNodesAtLevel(unsigned int *degrees, unsigned int V, unsigned int* buffer, unsigned int** helper, unsigned int* e, unsigned int level){
    unsigned int global_threadIdx = blockIdx.x * blockDim.x + threadIdx.x; 
    for(unsigned int i=global_threadIdx; i<V; i+= N_THREADS){
        // if(i>N_THREADS && THID == 50) printf("%d:%d ", blockIdx.x, i);
        if(degrees[i] == level){
            unsigned int loc = getWriteLoc(helper, e);
            writeToBuffer(buffer, helper, loc, i);
        }
    }
    __syncthreads();
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


    selectNodesAtLevel(d_p.degrees, V, buffer, &helper, &e, level);

    __syncthreads();
    if(THID==0 && level == 1) printf("%d ", e);


    // e is being incrmented within the loop, 
    // warps should process all the nodes added during the execution of loop
    // for that purpose e_processes is introduced, is incremented whenever a warp takes a job. 
    
    
    // for(unsigned int i = warp_id; i<e ; i = warp_id + base){
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
        
        
        unsigned int v, start, end;

        // only first lane reads buffer, start and end
        // it is then broadcasted to all lanes in the warp
        // it's done to reduce multiple accesses to global memory... 

        if(lane_id == 0){ 
            v = readFromBuffer(buffer, &helper, i);
            start = d_p.neighbors_offset[v];
            end = d_p.neighbors_offset[v+1];
        }

        v = __shfl_sync(0xFFFFFFFF, v, 0);
        start = __shfl_sync(0xFFFFFFFF, start, 0);
        end = __shfl_sync(0xFFFFFFFF, end, 0);

        for(int j = start + lane_id; j<end ; j+=32){
            unsigned int u = d_p.neighbors[j];
            if(d_p.degrees[u] > level){
                unsigned int a = atomicSub(d_p.degrees+u, 1);
            
                if(a == level+1){
                    unsigned int loc = getWriteLoc(&helper, &e);
                    writeToBuffer(buffer, &helper, loc, u);
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


