
#include "../inc/device_funcs.h"
#include "stdio.h"
__device__ void exclusiveScanWarpLevel(unsigned int* addresses){
    int lane_id = THID%32;

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
}
__device__ void compactWarpLevel(unsigned int *degrees, unsigned int V, unsigned int* w_buffer, 
                                    unsigned int** w_helper, unsigned int* w_e, unsigned int level){

    unsigned int warp_id = THID/32;
    unsigned int lane_id = THID%32;
    unsigned int global_threadIdx = blockIdx.x * blockDim.x + THID; 
    __shared__ bool predicate[BLK_DIM];
    __shared__ unsigned int addresses[BLK_DIM];

    
    for(unsigned int base = 0; base < V; base += N_THREADS){
        
        unsigned int v = base + global_threadIdx; 

        // all threads should get some value, if vertices are less rest of the threads get zero
        predicate[THID] = v<V? (degrees[v] == level) : 0;

        addresses[THID] = predicate[THID];

        exclusiveScanWarpLevel(addresses+(warp_id*WARP_SIZE));

        

        if(     //check if we need to allocate a glBuffer for this warp
            (lane_id == WARP_SIZE-1) && // only one thread in a warp does this job
                // w_e: no. of nodes already selected, addresses[...]: no. of nodes in currect scan
            (w_e[0] + addresses[THID] >= MAX_NV) &&  
                // check if it's not already allocated
            (w_helper[0] == NULL)
            ){

            w_helper[0] = (unsigned int*) malloc(sizeof(unsigned int) * GLBUFFER_SIZE);
            printf("allocated%d ", *w_e + addresses[THID]);
            }
        __syncwarp();
        
        if(predicate[THID]){
            unsigned int loc = addresses[THID] + w_e[0];
            writeToBuffer(w_buffer, w_helper, loc, v);
        }

        // if(global_threadIdx > 31 && global_threadIdx<96)
        //     printf("%d-%d ", addresses[THID], w_buffer[loc]);
        __syncwarp();

        if(lane_id == WARP_SIZE - 1){
            // atomicAdd(w_e, addresses[THID]);
            w_e[0] += (addresses[THID] + predicate[THID]);
        }

        __syncwarp();
 
    }
}

__device__ unsigned int getWriteLoc(unsigned int** glBuffer, unsigned int* bufTail){
    unsigned int loc = atomicAdd(bufTail, 1);
    assert(loc < GLBUFFER_SIZE + MAX_NV);

    if(loc == MAX_NV){ // checking equal so that only one thread in a warp should allocate glBuffer
        glBuffer[0] = (unsigned int*) malloc(sizeof(unsigned int) * GLBUFFER_SIZE); 
        printf("Memory allocate in atomic");  
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

__device__ unsigned int readFromBuffer(unsigned int* shBuffer, unsigned int** glBuffer, unsigned int loc){
    assert(loc < MAX_NV + GLBUFFER_SIZE);
    return ( loc < MAX_NV ) ? shBuffer[loc] : glBuffer[0][loc-MAX_NV]; 
}

__global__ void PKC(G_pointers d_p, unsigned int *global_count, int level){


    __shared__ unsigned int shBuffer[WARPS_EACH_BLK*MAX_NV];
    __shared__ unsigned int bufTail[WARPS_EACH_BLK];
    __shared__ unsigned int* helpers[WARPS_EACH_BLK];


    unsigned int warp_id = THID/32;
    unsigned int lane_id = THID%32;

  //  unsigned int global_idx = (blockIdx.x)*WARPS_EACH_BLK+warp_id;
  //  unsigned int mask = 0xFFFFFFFF;

    if(lane_id==0){
        bufTail[warp_id] = 0;
        helpers[warp_id] = NULL;
    }
	


    compactWarpLevel(d_p.degrees, d_p.V, shBuffer+(warp_id*MAX_NV), helpers+warp_id, bufTail+warp_id, level);

    __syncwarp();

    for(unsigned int i=0; i < bufTail[warp_id]; i++){
        unsigned int v, start, end;
        if(lane_id == 0){ 
            v = readFromBuffer(shBuffer+(warp_id*MAX_NV), helpers+warp_id, i);
            start = d_p.neighbors_offset[v];
            end = d_p.neighbors_offset[v+1];
        }        
        v = __shfl_sync(0xFFFFFFFF, v, 0);
        start = __shfl_sync(0xFFFFFFFF, start, 0);
        end = __shfl_sync(0xFFFFFFFF, end, 0);
        
        for(int j = start + lane_id; j<end ; j += WARP_SIZE){
            unsigned int u = d_p.neighbors[j];
            if(d_p.degrees[u] > level){
                unsigned int a = 0;
                a = atomicSub(d_p.degrees+u, 1);
            
                if(a == (level+1)){
                    // node degree became the level after decrementing... 
                    // this node should be processsed at this level, hence should be added to shBuffer/glBuffer
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
        atomicAdd(global_count, bufTail[warp_id]);
        if(helpers[warp_id]!=NULL) free(helpers[warp_id]);  
	}

}

