#include "../inc/scans.h"
#include "../inc/common.h"

__device__ void scanBlockHillis(unsigned int* addresses){
    // Hillis Steele Scan
    // todo check this code is working
    
    int initVal = addresses[THID];

    for (unsigned int d = 1; d < BLK_DIM; d = d*2) {
        unsigned int newVal = addresses[THID];   
        if (int(THID - d) >= 0)  
            newVal += addresses[THID-d];  
        __syncthreads();  
        addresses[THID] = newVal;
    }
        //Hillis-Steele Scan gives inclusive scan.
        //to get exclusive scan, subtract the initial values.
    addresses[THID] -= initVal;
    __syncthreads();
}

__device__ void scanBlockBelloch(unsigned int* addresses){

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
    __syncthreads();
}

__device__ void scanWarpHillis(unsigned int* addresses){
    int initVal = addresses[THID];
    
    for (unsigned int d = 1; d < WARP_SIZE; d = d*2) {
        unsigned int newVal = addresses[THID];   
        if (int(THID - d) >= 0)  
            newVal += addresses[THID-d];  
        __syncwarp();  
        addresses[THID] = newVal;
    }
        //Hillis-Steele Scan gives inclusive scan.
        //to get exclusive scan, subtract the initial values.
    addresses[THID] -= initVal;
    __syncwarp();
}

__device__ void scanWarpBelloch(unsigned int* addresses){
    unsigned int lane_id = THID%WARP_SIZE;
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
    __syncwarp();
}
