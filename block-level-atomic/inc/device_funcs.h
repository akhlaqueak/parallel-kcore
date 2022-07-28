

#ifndef CUTS_DEVICE_FUNCS_H
#define CUTS_DEVICE_FUNCS_H
#include "./common.h"


__device__ void selectNodesAtLevel(unsigned int *degrees, unsigned int V, unsigned int* w_buffer, unsigned int** helpers, unsigned int* e, unsigned int level);

__device__ void writeToBuffer(unsigned int* w_buffer,  unsigned int** w_helper, unsigned int* w_e, unsigned int v);

__global__ void PKC(G_pointers d_p, unsigned int *global_count, int level, int V);

__device__ unsigned int readFromBuffer(unsigned int* w_buffer, unsigned int** w_helper, unsigned int loc);

#endif //CUTS_DEVICE_FUNCS_H
