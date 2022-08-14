#include "../inc/scans.h"


__device__ void scanBlockHillis(unsigned int* addresses){
    // Hillis Steele Scan
    // todo check this code is working
    __syncthreads();
    int initVal = addresses[THID];

    for (unsigned int d = 1; d < BLK_DIM; d = d*2) {
        unsigned int newVal = addresses[THID];   
        if (int(THID - d) >= 0)  
            newVal += addresses[THID-d];  
        __syncthreads();
        addresses[THID] = newVal;
        __syncthreads();  
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
    __syncwarp();  
    int lane_id = THID%32;
    int initVal = addresses[lane_id];

    for (unsigned int d = 1; d < WARP_SIZE; d = d*2) {
        unsigned int newVal = addresses[lane_id];   
        if (int(lane_id - d) >= 0)  
            newVal += addresses[lane_id-d];  
        __syncwarp();  
        addresses[lane_id] = newVal;
        __syncwarp();  
    }
        //Hillis-Steele Scan gives inclusive scan.
        //to get exclusive scan, subtract the initial values.
    addresses[lane_id] -= initVal;
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

__device__ void compactBlock(unsigned int *degrees, unsigned int V, Node** tail, Node** head, unsigned int* bufTailPtr, unsigned int level){

    unsigned int glThreadIdx = blockIdx.x * BLK_DIM + THID; 
    __shared__ bool predicate[BLK_DIM];
    __shared__ unsigned int addresses[BLK_DIM];
    __shared__ unsigned int bTail;
    
    for(unsigned int base = 0; base < V; base += N_THREADS){
        
        unsigned int v = base + glThreadIdx; 

        // all threads should get some value, if vertices are less than n_threads, rest of the threads get zero
        predicate[THID] = (v<V)? (degrees[v] == level) : 0;

        addresses[THID] = predicate[THID];

        scanBlock(addresses);

        
        if(THID == BLK_DIM - 1){  
            int nv =  addresses[THID] + predicate[THID];            
            bTail = nv>0? atomicAdd(bufTailPtr, nv) : 0;
            
            if(allocationRequired(tail[0], bTail+nv)) // adding nv since bTail is old value of bufTail
                allocateMemory(tail, head);
        }

        // this sync is necessary so that memory is allocated before writing to buffer
        __syncthreads();
        
        addresses[THID] += bTail;
        
        if(predicate[THID])
            writeToBuffer(tail[0], addresses[THID], v);
        
        __syncthreads();
            
    }
}

__device__ void compactWarp(unsigned int* temp, unsigned int* addresses, unsigned int* predicate, 
                            Node** tail, Node** head, unsigned int* bufTailPtr, 
                            volatile unsigned int* lock){
    
    // __syncwarp();

    unsigned int lane_id = THID%WARP_SIZE;

    unsigned int bTail;
    
    addresses[lane_id] = predicate[lane_id];

    scanWarp(addresses);
    // todo: look for atomic add at warp level.
    
    if(lane_id == WARP_SIZE-1){
        int nv = addresses[lane_id]+predicate[lane_id];
        bTail = nv>0? atomicAdd(bufTailPtr, nv) : 0;
        if(allocationRequired(tail[0], bTail+nv)){ // adding nv since bTail is old value of bufTail
            // atomicCAS((unsigned int*)&lock, 2, 0); // resets the lock in case a memory was allocated before
            // __threadfence_block();
            printf("Req %d", THID);
            allocateMemoryMutex(tail, head, lock);
        }   
    }  
    
    bTail = __shfl_sync(0xFFFFFFFF, bTail, WARP_SIZE-1);
    
    addresses[lane_id] += bTail;

    if(predicate[lane_id])
        writeToBuffer(tail[0], addresses[lane_id], temp[lane_id]);

    // reset for next iteration
    predicate[lane_id] = 0;

    // __syncwarp();
}
