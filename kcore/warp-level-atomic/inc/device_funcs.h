

#ifndef CUTS_DEVICE_FUNCS_H
#define CUTS_DEVICE_FUNCS_H
#include "./common.h"


__device__ void selectNodesAtLevel(unsigned int *degrees, unsigned int V, unsigned int* w_buffer, unsigned int** helpers, unsigned int* e, unsigned int level);

__global__ void PKC(G_pointers d_p, unsigned int *global_count, int level, int V);

__device__ unsigned int readFromBuffer(unsigned int* w_buffer, unsigned int** w_helper, unsigned int loc);

__device__ unsigned int getWriteLoc(unsigned int** helper, unsigned int* e);

__device__ void writeToBuffer(unsigned int* buffer,  unsigned int** helper, unsigned int loc, unsigned int v);
#endif //CUTS_DEVICE_FUNCS_H
