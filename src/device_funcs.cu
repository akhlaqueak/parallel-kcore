
#include "../inc/device_funcs.h"
#include "stdio.h"


__device__ void scan(unsigned int *degrees, unsigned int V, unsigned int* w_buffer, unsigned int** helpers, unsigned int* e, unsigned int level){
    unsigned int warp_id = threadIdx.x/32;
    unsigned int global_threadIdx = blockIdx.x * blockDim.x + threadIdx.x; 
    for(unsigned int i=global_threadIdx; i< V; i+= N_THREADS){
        if(degrees[i] == level){
		if(e[warp_id] >= MAX_NV){
            printf("x"); continue;
        }

            //store this node to shared buffer, at the corresponding warp location
            unsigned int loc = atomicAdd(&e[warp_id], 1);

            if(loc == MAX_NV){
                helpers[warp_id] = (unsigned int*) malloc(HELPER_SIZE);
            }

            if(loc >= MAX_NV){
                loc-= MAX_NV;
                helpers[warp_id][loc] = i;
            }

            else{
                w_buffer[loc] = i;
            }
        }
    }
}

__global__ void PKC(G_pointers d_p, unsigned int *global_count, int level, int V){


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
	

    __syncwarp();

    scan(d_p.degrees, V, &buffer[warp_id*MAX_NV], e, level);


    for(int i=0; i<e[warp_id]; i++){
    
        unsigned int v;
        if( i < MAX_NV ) 
            v = buffer[warp_id*MAX_NV + i];
        else
            v = helpers[warp_id][i-MAX_NV];


        unsigned int start = d_p.neighbors_offset[v];
        unsigned int end = d_p.neighbors_offset[v+1];

        for(int j = start + lane_id; j<end ; j+=32){
            unsigned int u = d_p.neighbors[j];
            if(d_p.degrees[u] > level){
                unsigned int a = 0;
                a = atomicSub(&d_p.degrees[u], 1);
            
                if(a == (level+1)){
        // node degree became the level after decrementing... 
                    unsigned int loc = atomicAdd(&e[warp_id], 1); 

                    
                    if(loc == MAX_NV){
                        helpers[warp_id] = (unsigned int*) malloc(HELPER_SIZE);
                    }

                    if(loc >= MAX_NV){
                        loc-= MAX_NV;
                        helpers[warp_id][loc] = i;
                    }

                    else{
                        buffer[warp_id*MAX_NV + loc] = i;
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
        if(helpers[warp_id]!=NULL)
            free(helpers[warp_id]);  
	}

}


