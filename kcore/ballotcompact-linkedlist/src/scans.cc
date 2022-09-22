#include "../inc/common.h"

enum{INCLUSIVE, EXCLUSIVE};
__device__ unsigned int scanWarpHellis(volatile unsigned int* addresses, unsigned int type){
    const unsigned int lane_id = THID & 31;

    for(int i=1; i<WARP_SIZE; i*=2){
        if(lane_id >= i)
            addresses[THID] += addresses[THID-i];
    }

    
    if(type == INCLUSIVE)
        return addresses[THID];
    else{
        return (lane_id>0)? addresses[THID-1] : 0;
    }    
}

__device__ unsigned int scanWarpBallot(volatile unsigned int* addresses, unsigned int type){
    uint lane_id = THID & 31;
    uint bits = __ballot_sync(0xffffffff, addresses[THID]);
    uint mask = 0xffffffff >> (31-lane_id);
    addresses[THID] = __popc(mask & bits);
    if(type == INCLUSIVE)
        return addresses[THID];
    else
        return lane_id>0? addresses[THID-1] : 0;
}



__device__ void scanBlock(volatile unsigned int* addresses, unsigned int type){
    const unsigned int lane_id = THID & 31;
    const unsigned int warp_id = THID >> 5;
    
    unsigned int val = scanWarpBallot(addresses, type);
    __syncthreads();

    if(lane_id==31)
        addresses[warp_id] = addresses[THID];
    __syncthreads();

    if(warp_id==0)
    // it can't be ballot scan as elements are no more binary
        scanWarpHellis(addresses, INCLUSIVE);
    __syncthreads();

    if(warp_id>0)
        val += addresses[warp_id-1];
    __syncthreads();

    addresses[THID] = val;
    __syncthreads();
    
}





__device__ void compactWarp(bool* predicate, volatile unsigned int* addresses, unsigned int* temp, 
                            unsigned int* shBuffer, Node** tail, Node** head, unsigned int* bufTailPtr, 
                            volatile unsigned int* lock, unsigned int* total){
    
    // __syncwarp();

    unsigned int lane_id = THID & 31;

    unsigned int bTail;
    
    addresses[THID] = predicate[THID];

    unsigned int address = scanWarpBallot(addresses, EXCLUSIVE);
    // todo: look for atomic add at warp level.
    
    if(lane_id == WARP_SIZE-1){
        unsigned int nv = address + predicate[THID]; // nv can be zero if no vertex was found in this warp
        bTail = nv>0? atomicAdd(bufTailPtr, nv) : 0; // don't want to do atomicAdd for zero
        if(allocationRequired(tail[0], bTail+nv)){ // adding nv since bTail is old value of bufTail
            // printf("Req %d", THID);
            // atomicCAS((unsigned int*)lock, 2, 0); // resets the lock in case a memory was allocated before
            // __threadfence_block();   //with atomic operations it's not required.
            allocateMemoryMutex(tail, head, lock, total);
        }   
    }  
    
    bTail = __shfl_sync(0xFFFFFFFF, bTail, WARP_SIZE-1);
    
    address += bTail;
    
    if(predicate[THID])
        writeToBuffer(shBuffer, tail[0], address, temp[THID]);
        
        // reset for next iteration
    predicate[THID] = 0;

        
    // __syncwarp();
}


__device__ void compactBlock(bool* predicate, volatile unsigned int* addresses, unsigned int* temp,
    unsigned int* shBuffer, Node** tail, Node** head, unsigned int* bufTailPtr, unsigned int* total){


    __shared__ unsigned int bTail;
    
    addresses[THID] = predicate[THID];

    scanBlock(addresses, EXCLUSIVE);

    
    if(THID == BLK_DIM - 1){  
        int nv =  addresses[THID] + predicate[THID];            
        bTail = nv>0? atomicAdd(bufTailPtr, nv) : 0;
        
        if(allocationRequired(tail[0], bTail+nv)){ // adding nv since bTail is old value of bufTail
            allocateMemory(tail, head);
            atomicAdd(total, 1); // total is only for reporting how many nodes created
        }
    }

    // this sync is necessary so that memory is allocated before writing to buffer
    __syncthreads();
    
    addresses[THID] += bTail;
    
    if(predicate[THID])
        writeToBuffer(shBuffer, tail[0], addresses[THID], temp[THID]);
    
    __syncthreads();
            
}