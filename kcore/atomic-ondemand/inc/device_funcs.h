

#ifndef CUTS_DEVICE_FUNCS_H
#define CUTS_DEVICE_FUNCS_H
#include "./common.h"

__device__ void selectNodesAtLevel(unsigned int *degrees, unsigned int V, unsigned int* shBuffer, unsigned int** glBuffer, unsigned int* bufTail, unsigned int level);

__device__ void syncBlocks(volatile unsigned int* blockCounter);

__global__ void PKC(G_pointers d_p, unsigned int *global_count, int level, int V, volatile unsigned int* blockCounter);

#endif //CUTS_DEVICE_FUNCS_H
