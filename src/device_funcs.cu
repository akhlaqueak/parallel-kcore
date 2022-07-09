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


__device__ void scan(G_pointers d_p, unsigned int* buffer, unsigned int* e, unsigned int level){
    unsigned int warp_id = threadIdx.x/32;
    unsigned int lane_id = threadIdx.x%32;
    unsigned int global_threadIdx = blockIdx.x*BLK_DIM + threadIdx.x; 

    for(int i=global_threadIdx; i<d_p.V; i+=N_THREADS){
	printf("%d, %d, %d", global_threadIdx, level, d_p.degrees[i]);
        if(d_p.degrees[i] == level){
		printf("degree matched \n");
            //store this node to shared buffer, at the corresponding warp location
            unsigned int loc = warp_id*MAX_NE + e[warp_id]; 
            buffer[loc] = i;
            atomicAdd(&e[warp_id], 1); 
        }
    }
}

__global__ void PKC(G_pointers d_p, unsigned int *global_count, int level){


    __shared__ unsigned int buffer[WARPS_EACH_BLK*MAX_NE];
    __shared__ unsigned int e[WARPS_EACH_BLK];



    unsigned int warp_id = threadIdx.x/32;
    unsigned int lane_id = threadIdx.x%32;
    unsigned int global_idx = (blockIdx.x)*WARPS_EACH_BLK+warp_id;
    unsigned int mask = 0xFFFFFFFF;

    if(lane_id==0)
        e[warp_id] = 0;

    __syncwarp();
    scan(d_p, buffer, e, level);
    __syncthreads();

    for(int i=0; i<e[warp_id]; i++){
        unsigned int v = buffer[i];
        unsigned int start = d_p.neighbors_offset[v+1];
        unsigned int end = d_p.neighbors_offset[v];
        for(int j = 0; j<(end - start); j+=32){
            int a = 0;
            unsigned int u = d_p.neighbors[j];
            if(d_p.degrees[u] > level){
                a = atomicSub(&d_p.degrees[u], 1);
		printf("subtracted\n");
            }

            if(a == (level+1)){
                int loc = warp_id*MAX_NE + e[warp_id];
                buffer[loc] = u;
                atomicAdd(&e[warp_id], 1);
            }

            if(a <= level){
                atomicAdd(&d_p.degrees[u], 1);
            }
        }

        __syncwarp();
    }

    if(lane_id == 0){
        atomicAdd(global_count, e[warp_id]);    
        printf("global count: %d, was added with %d\n", global_count[0], e[warp_id]); 
	}
}
