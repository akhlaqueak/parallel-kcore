#ifndef BUFFER_H
#define BUFFER_H

#include "./common.h"

__device__ void writeToBuffer(unsigned int* shBuffer,    unsigned int* glBuffer, unsigned int loc, unsigned int v);

__device__ unsigned int readFromBuffer(unsigned int* shBuffer,  unsigned int* glBuffer, unsigned int loc);

__device__ inline bool allocationRequired( unsigned int* glBuffer, unsigned int loc, unsigned int dim);

__device__ inline void allocateMemory( unsigned int** glBufferPtr);

__device__ void allocateMemoryMutex( unsigned int** glBufferPtr, unsigned int loc, volatile unsigned int* lock);


#endif //BUFFER_H