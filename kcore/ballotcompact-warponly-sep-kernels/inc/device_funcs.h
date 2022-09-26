

#ifndef CUTS_DEVICE_FUNCS_H
#define CUTS_DEVICE_FUNCS_H
#include "./common.h"


#include "../inc/device_funcs.h"
#include "stdio.h"

__global__ void selectNodesAtLevel(unsigned int* degrees, unsigned int *bufTails, int level, int V, 
    unsigned int* glBuffers);
    
__global__ void processNodes(G_pointers d_p, unsigned int *global_count, int level, int V, 
                    unsigned int* bufTails, unsigned int* glBuffers);
  
#endif //CUTS_DEVICE_FUNCS_H
