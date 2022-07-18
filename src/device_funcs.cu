/*
 * cuTS:  Scaling Subgraph Isomorphism on Distributed Multi-GPU Systems Using
 *        Trie Based Data Structure
 *
 * Copyright (C) 2021 APPL Laboratories (aravind_sr@outlook.com)
 *
 * This software is available under the MIT license, a copy of which can be
 * found in the file 'LICENSE' in the top-level directory.
 *
 * For further information contact:
 *   (1) Lizhi Xiang (lizhi.xiang@wsu.edu)
 *   (2) Aravind Sukumaran-Rajam (aravind_sr@outlook.com)
 *
 * The citation information is provided in the 'README' in the top-level
 * directory.
 */
#include "../inc/device_funcs.h"
#include "stdio.h"


__device__ void scan(unsigned int *degrees, unsigned int V, unsigned int* buffer, unsigned int* e, unsigned int level){
    unsigned int warp_id = threadIdx.x/32;
    unsigned int global_threadIdx = blockIdx.x * blockDim.x + threadIdx.x; 
    for(unsigned int i=global_threadIdx; i< V; i+= N_THREADS){
        if(degrees[i] == level){
		if(e[warp_id] >= MAX_NV){
            printf("x"); continue;
        }

            //store this node to shared buffer, at the corresponding warp location
            unsigned int loc = atomicAdd(&e[warp_id], 1); 
            loc = loc + warp_id*MAX_NV; 
            buffer[loc] = i;
        }
    }
}

__global__ void PKC(G_pointers d_p, unsigned int *global_count, int level, int V){


    __shared__ unsigned int buffer[WARPS_EACH_BLK*MAX_NV];
    __shared__ unsigned int e[WARPS_EACH_BLK];



    unsigned int warp_id = threadIdx.x/32;
    unsigned int lane_id = threadIdx.x%32;

  //  unsigned int global_idx = (blockIdx.x)*WARPS_EACH_BLK+warp_id;
  //  unsigned int mask = 0xFFFFFFFF;

    if(lane_id==0)
        e[warp_id] = 0;
	

    __syncwarp();

    scan(d_p.degrees, V, buffer, e, level);


    for(int i=0; i<e[warp_id]; i++){
    
        unsigned int v = buffer[warp_id*MAX_NV + i];
        unsigned int start = d_p.neighbors_offset[v];
        unsigned int end = d_p.neighbors_offset[v+1];

        for(int j = start + lane_id; j<end ; j+=32){
            unsigned int u = d_p.neighbors[j];
            if(d_p.degrees[u] > level){
                unsigned int a = 0;
                a = atomicSub(&d_p.degrees[u], 1);
            
                if(a == (level+1)){
        
                    if(e[warp_id] >= MAX_NV){
                        printf("x");
                    }
                    unsigned int loc = atomicAdd(&e[warp_id], 1); 
                    loc = loc + warp_id*MAX_NV;
                    buffer[loc] = u;           
                    // printf("**u=%d*i=%d*v=%d**", u, i, v);
                }

                if(a <= level){
                    // printf("%d+", u);
                    atomicAdd(&d_p.degrees[u], 1);
                }
            }
        }

        __syncwarp();
    }


    if(lane_id == 0 && e[warp_id]!=0 ){
        atomicAdd(&global_count[0], e[warp_id]);    
	}

}


