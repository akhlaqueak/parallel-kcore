
#include "../inc/device_funcs.h"
#include "stdio.h"
__device__ void exclusiveScanWarpLevel(unsigned int* addresses){
    int lane_id = threadIdx.x%32;

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
    unsigned int warp_id = threadIdx.x/32;
    unsigned int global_threadIdx = blockIdx.x * blockDim.x + threadIdx.x; 
    __shared__ bool predicate[BLK_DIM];
    __shared__ unsigned int addresses[BLK_DIM];
    unsigned int lane_id = threadIdx.x%32;
    
    for(unsigned int i = 0; i < V; i+= N_THREADS){
        
        unsigned int v = i + global_threadIdx; 

        // all threads should get some value, if vertices are less rest of the threads get zero
        predicate[threadIdx.x] = v<V? (degrees[v] == level) : 0;

        addresses[threadIdx.x] = predicate[threadIdx.x];

        exclusiveScanWarpLevel(&addresses[warp_id*WARP_SIZE]);

        if(     //check if we need to allocate a helper for this warp
            (lane_id == WARP_SIZE-1) && // only one thread in a warp does this job
                // w_e: no. of nodes already selected, addresses[...]: no. of nodes in currect scan
            (*w_e + addresses[threadIdx.x] >= MAX_NV) &&  
                // check if it's not already allocated
            (*w_helper == NULL)
            ){

            *w_helper = (unsigned int*) malloc(HELPER_SIZE);
            printf("allocated%d ", *w_e + addresses[threadIdx.x]);
            }
        __syncwarp();
        
            unsigned int loc = 0;
        if(predicate[threadIdx.x]){
            loc = addresses[threadIdx.x] + w_e[0];
            if(loc < MAX_NV)
                w_buffer[loc] = v;
            else
                *(*w_helper + (loc - MAX_NV)) = v;            
        }

        if(global_threadIdx > 31 && global_threadIdx<96)
            printf("%d-%d ", addresses[threadIdx.x], w_buffer[loc]);
        __syncwarp();

        if(lane_id == WARP_SIZE - 1){
            // atomicAdd(w_e, addresses[threadIdx.x]);
            w_e[0] += addresses[threadIdx.x];
        }

        __syncwarp();
 
    }
}

__global__ void PKC(G_pointers d_p, unsigned int *global_count, int level){


    __shared__ unsigned int buffer[WARPS_EACH_BLK*MAX_NV];
    __shared__ unsigned int e[WARPS_EACH_BLK];
    __shared__ unsigned int* helpers[WARPS_EACH_BLK];


    unsigned int warp_id = threadIdx.x/32;
    unsigned int lane_id = threadIdx.x%32;

  //  unsigned int global_idx = (blockIdx.x)*WARPS_EACH_BLK+warp_id;
  //  unsigned int mask = 0xFFFFFFFF;

    if(lane_id==0){
        e[warp_id] = 0;
        helpers[warp_id] = NULL;
    }
	


    compactWarpLevel(d_p.degrees, d_p.V, &buffer[warp_id*MAX_NV], &helpers[warp_id], &e[warp_id], level);

    __syncwarp();

    for(unsigned int i=0; i < e[warp_id]; i++){
    
        unsigned int v;
        if( i < MAX_NV ) 
            v = buffer[warp_id*MAX_NV + i];
        else
            v = *(helpers[warp_id] + (i-MAX_NV));


        unsigned int start = d_p.neighbors_offset[v];
        unsigned int end = d_p.neighbors_offset[v+1];

        for(int j = start + lane_id; j<end ; j += WARP_SIZE){
            unsigned int u = d_p.neighbors[j];
            if(d_p.degrees[u] > level){
                unsigned int a = 0;
                a = atomicSub(&d_p.degrees[u], 1);
            
                if(a == (level+1)){
        // node degree became the level after decrementing... 
                    unsigned int loc = atomicAdd(&e[warp_id], 1); 
                    if(e[warp_id] >= HELPER_SIZE){
                        printf("x"); continue;
                    }
                    
                    if(loc == MAX_NV){
                        helpers[warp_id] = (unsigned int*) malloc(HELPER_SIZE);
                        if(helpers[warp_id] == NULL) printf("Memory not allocated... ");
                        // else printf("success.");
                        __syncwarp();
                    }

                    if(loc < MAX_NV){

                        buffer[warp_id*MAX_NV + loc] = u;
                        // printf("%dz", loc);
                    }

                    else{
                        loc-= MAX_NV;
                        *(helpers[warp_id] + loc) = u;
                    }          
                }

                if(a <= level){
        // node degree became less than the level after decrementing... 
                    atomicAdd(&d_p.degrees[u], 1);
                }
            }
        }

        __syncwarp();
    }

    if(lane_id == 0 && e[warp_id]!=0 ){
        atomicAdd(&global_count[0], e[warp_id]);
        free(helpers[warp_id]);  
	}

}


