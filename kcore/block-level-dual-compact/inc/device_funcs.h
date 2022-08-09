

#ifndef CUTS_DEVICE_FUNCS_H
#define CUTS_DEVICE_FUNCS_H
#include "./common.h"

#include "../inc/device_funcs.h"
#include "stdio.h"



__device__ void writeToBuffer(unsigned int* shBuffer,    unsigned int* glBuffer, unsigned int loc, unsigned int v);

__device__ unsigned int readFromBuffer(unsigned int* shBuffer,  unsigned int* glBuffer, unsigned int loc);

__device__ inline bool allocationRequired( unsigned int* glBuffer, unsigned int loc, unsigned int dim);

__device__ inline void allocateMemory( unsigned int** glBufferPtr);

__device__ void allocateMemoryMutex( unsigned int** glBufferPtr, unsigned int loc, volatile unsigned int* lock);

__global__ void PKC(G_pointers d_p, unsigned int *global_count, int level, int V, volatile unsigned int* blockCounter);

__device__ void syncBlocks(volatile unsigned int* blockCounter);

#endif //CUTS_DEVICE_FUNCS_H
