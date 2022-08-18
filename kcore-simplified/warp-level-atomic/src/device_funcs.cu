
#include "../inc/device_funcs.h"
#include "stdio.h"
__device__ unsigned int getWriteLoc(unsigned int** glBuffer, unsigned int* bufTail){
    unsigned int loc = atomicAdd(bufTail, 1);
    assert(loc < GLBUFFER_SIZE + MAX_NV);

    if(loc == MAX_NV){ // checking equal so that only one thread in a warp should allocate glBuffer
        glBuffer[0] = (unsigned int*) malloc(sizeof(unsigned int) * GLBUFFER_SIZE); 
        assert(glBuffer[0] != NULL); 
    }
    return loc;
}

__device__ void writeToBuffer(unsigned int* shBuffer,  unsigned int** glBuffer, unsigned int loc, unsigned int v){
    assert(loc < GLBUFFER_SIZE + MAX_NV);
    if(loc < MAX_NV){
        shBuffer[loc] = v;
    }
    else{
        assert(glBuffer[0]!=NULL);
        glBuffer[0][loc-MAX_NV] = v; 
    }
}

__device__ void selectNodesAtLevel(unsigned int *degrees, unsigned int V, unsigned int* w_buffer, unsigned int** w_helper, unsigned int* w_e, unsigned int level){
    unsigned int global_threadIdx = blockIdx.x * blockDim.x + threadIdx.x; 
    for(unsigned int i=global_threadIdx; i< V; i+= N_THREADS){
        if(degrees[i] == level){
            unsigned int loc = getWriteLoc(w_helper, w_e);
            writeToBuffer(w_buffer, w_helper, loc, i);
        }
    }
}

__device__ unsigned int readFromBuffer(unsigned int* shBuffer, unsigned int** glBuffer, unsigned int loc){
    assert(loc < MAX_NV + GLBUFFER_SIZE);
    return ( loc < MAX_NV ) ? shBuffer[loc] : glBuffer[0][loc-MAX_NV]; 
}


__global__ void PKC(G_pointers d_p, unsigned int *global_count, int level, int V){


    __shared__ unsigned int shBuffer[WARPS_EACH_BLK*MAX_NV];
    __shared__ unsigned int bufTail[WARPS_EACH_BLK];
    __shared__ unsigned int* helpers[WARPS_EACH_BLK];


    unsigned int warp_id = threadIdx.x/32;
    unsigned int lane_id = threadIdx.x%32;

    if(lane_id==0){
        bufTail[warp_id] = 0;
        helpers[warp_id] = NULL;
    }
	
    // TODO: remove the warp level implementations, go to block level.
    __syncthreads();

    selectNodesAtLevel(d_p.degrees, V, shBuffer+warp_id*MAX_NV, helpers+warp_id, bufTail+warp_id, level);

    __syncthreads();

    // this loop is good, as every thread in a warp will run if i<bufTail[warp_id]
    for(unsigned int i=0; i<bufTail[warp_id]; i++){
        unsigned int v, start, end;
        if(lane_id == 0){ 
            v = readFromBuffer(shBuffer+(warp_id*MAX_NV), helpers+warp_id, i);
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
                    unsigned int loc = getWriteLoc(helpers+warp_id, bufTail+warp_id);
                    writeToBuffer(shBuffer+(warp_id*MAX_NV), helpers+warp_id, loc, u);
                }

                if(a <= level){
                    // node degree became less than the level after decrementing... 
                    atomicAdd(d_p.degrees+u, 1);
                }
            }
        }

        __syncwarp();
    }

    if(lane_id == 0 && bufTail[warp_id]!=0 ){
        atomicAdd(global_count, bufTail[warp_id]); //: global_count only can be replaced
        if(helpers[warp_id]) free(helpers[warp_id]);  //TODO: check if glBuffer is allotted... 
	}

}


