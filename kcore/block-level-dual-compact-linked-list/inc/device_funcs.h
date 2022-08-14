

#ifndef CUTS_DEVICE_FUNCS_H
#define CUTS_DEVICE_FUNCS_H
#include "./common.h"
#include "./buffer.h"
#include "./scans.h"

__global__ void PKC(G_pointers d_p, unsigned int *global_count, int level, int V, volatile unsigned int* blockCounter);

__device__ void syncBlocks(volatile unsigned int* blockCounter);

#endif //CUTS_DEVICE_FUNCS_H