

#ifndef CUTS_DEVICE_FUNCS_H
#define CUTS_DEVICE_FUNCS_H
#include "./common.h"


__device__ void scan(unsigned int *degrees, unsigned int V, unsigned int* w_buffer, unsigned int** helpers, unsigned int* e, unsigned int level);

__global__ void PKC(G_pointers d_p, unsigned int *global_count, int level, int V);


#endif //CUTS_DEVICE_FUNCS_H
