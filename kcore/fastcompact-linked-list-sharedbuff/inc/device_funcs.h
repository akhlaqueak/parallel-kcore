

#ifndef CUTS_DEVICE_FUNCS_H
#define CUTS_DEVICE_FUNCS_H
#include "./common.h"
#include "./buffer.h"
__device__ void selectNodesAtLevel(bool* predicate, volatile unsigned int* addresses, unsigned int* temp,
    unsigned int *degrees, unsigned int V, unsigned int* shBuffer, Node** tail, Node** head, unsigned int* bufTail, unsigned int level);

__global__ void PKC(G_pointers d_p, unsigned int *global_count, int level, int V, volatile unsigned int* blockCounter);

__device__ void syncBlocks(volatile unsigned int* blockCounter);

#endif //CUTS_DEVICE_FUNCS_H
