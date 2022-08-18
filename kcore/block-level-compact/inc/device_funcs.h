

#ifndef CUTS_DEVICE_FUNCS_H
#define CUTS_DEVICE_FUNCS_H
#include "./common.h"


__device__ void selectNodesAtLevel(unsigned int *degrees, unsigned int V, unsigned int* shBuffer,  unsigned int* glBuffer, unsigned int* bufTail, unsigned int level, unsigned int* lock);

__device__ inline unsigned int getWriteLoc(unsigned int* bufTail);

__device__ void writeToBuffer(unsigned int* shBuffer,   unsigned int* glBuffer_p, unsigned int loc, unsigned int v, unsigned int* lock);

__device__ unsigned int readFromBuffer(unsigned int* shBuffer,  unsigned int* glBuffer, unsigned int loc);

__device__ void exclusiveScan(unsigned int* addresses);

__global__ void PKC(G_pointers d_p, unsigned int *global_count, int level, int V, unsigned int*);
#endif //CUTS_DEVICE_FUNCS_H
