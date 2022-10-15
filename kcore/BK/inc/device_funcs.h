

#ifndef CUTS_DEVICE_FUNCS_H
#define CUTS_DEVICE_FUNCS_H
#include "./common.h"

__global__ void selectNodesAtLevel(unsigned int *degrees, unsigned int level, unsigned int V, 
                 unsigned int* bufTails, unsigned int* glBuffers);


__global__ void processNodes(G_pointers d_p, int level, int V, 
                    unsigned int* bufTails, unsigned int* glBuffers, 
                    unsigned int *global_count);

__global__ void BK(G_pointers dp, Subgraphs* sg, unsigned int base);

#endif //CUTS_DEVICE_FUNCS_H
