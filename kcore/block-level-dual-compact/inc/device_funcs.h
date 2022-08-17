

#ifndef CUTS_DEVICE_FUNCS_H
#define CUTS_DEVICE_FUNCS_H
#include "./common.h"
#include "./buffer.h"
#include "./scans.h"

__global__ void PKC(G_pointers d_p, unsigned int *global_count, int level, int V, unsigned long long int* blockCounter);

__device__ void syncBlocks(unsigned long long int* blockCounter);

#endif //CUTS_DEVICE_FUNCS_H
