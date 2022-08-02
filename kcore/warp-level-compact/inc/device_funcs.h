

#ifndef CUTS_DEVICE_FUNCS_H
#define CUTS_DEVICE_FUNCS_H
#include "./common.h"


__device__ void compactWarpLevel(unsigned int *degrees, unsigned int V, unsigned int* w_buffer, 
                                    unsigned int** w_helper, unsigned int* w_e, unsigned int level);
__device__ void exclusiveScanWarpLevel(unsigned int* addresses);
__global__ void PKC(G_pointers d_p, unsigned int *global_count, int level);

__device__ unsigned int getWriteLoc(unsigned int** glBuffer, unsigned int* bufTail);
__device__ void writeToBuffer(unsigned int* shBuffer,  unsigned int** glBuffer, unsigned int loc, unsigned int v);

#endif //CUTS_DEVICE_FUNCS_H
