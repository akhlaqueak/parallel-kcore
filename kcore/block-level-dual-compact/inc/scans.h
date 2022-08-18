#ifndef SCANS_H
#define SCANS_H

#define BELLOCH false

#include "./common.h"
#include "./buffer.h"

__device__ void scanBlockHillis(unsigned int* addresses);

__device__ void scanBlockBelloch(unsigned int* addresses);

__device__ void scanWarpHillis(unsigned int* addresses);

__device__ void scanWarpBelloch(unsigned int* addresses);


__device__ inline void scanBlock(unsigned int* addresses){
    if(BELLOCH)
        scanBlockBelloch(addresses);
    else
        scanBlockHillis(addresses);
}

__device__ inline void scanWarp(unsigned int* addresses){
    if(BELLOCH)
        scanWarpBelloch(addresses);
    else
        scanWarpHillis(addresses);
}

// __device__ void compactBlock(unsigned int *degrees, unsigned int V, unsigned int* shBuffer,  
//      unsigned int** glBufferPtr, unsigned int* bufTailPtr, unsigned int level);
__device__ void compactBlock(unsigned int *degrees, unsigned int V, unsigned int* shBuffer,   
    unsigned int** glBufferPtr, unsigned int* bufTailPtr, unsigned int level);


__device__ void compactWarp(unsigned int* temp, unsigned int* addresses, unsigned int* predicate, 
    unsigned int* shBuffer,  unsigned int** glBufferPtr, unsigned int* bufTail_p, volatile unsigned int* lock);

#endif //SCANS_H
