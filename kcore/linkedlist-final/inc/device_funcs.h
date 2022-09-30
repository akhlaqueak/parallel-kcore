

#ifndef CUTS_DEVICE_FUNCS_H
#define CUTS_DEVICE_FUNCS_H
#include "./common.h"

__global__ void selectNodesAtLevel(unsigned int* degrees, unsigned int level, 
                        unsigned int V, unsigned int* bufTails,
                        Node** heads, Node** tails);
                        
__global__ void processNodes(G_pointers d_p, int level, int V, unsigned int* bufTails, 
    unsigned int* global_count, Node** heads, Node** tails);
#endif //CUTS_DEVICE_FUNCS_H
