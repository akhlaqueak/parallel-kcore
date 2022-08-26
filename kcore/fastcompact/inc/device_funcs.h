

#ifndef CUTS_DEVICE_FUNCS_H
#define CUTS_DEVICE_FUNCS_H
#include "./common.h"


#include "../inc/device_funcs.h"
#include "stdio.h"

__device__ void selectNodesAtLevel(bool* predicate, volatile unsigned int* addresses, unsigned int* temp,
    unsigned int *degrees, unsigned int V, unsigned int* shBuffer, unsigned int* glBuffer, unsigned int* bufTail, unsigned int level);

__device__ void syncBlocks(unsigned int* blockCounter);

__global__ void PKC(G_pointers d_p, unsigned int *global_count, int level, int V, 
                    unsigned int* blockCounter, unsigned int* glBuffers);


#endif //CUTS_DEVICE_FUNCS_H
