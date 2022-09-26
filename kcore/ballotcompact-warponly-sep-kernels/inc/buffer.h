#ifndef BUFFER_H
#define BUFFER_H

#include "./common.h"

__device__ unsigned int ldg (const unsigned int * p);

__device__ inline void writeToBuffer(unsigned int* glBuffer, unsigned int loc, unsigned int v);


__device__ inline unsigned int readFromBuffer( unsigned int* glBuffer, unsigned int loc);


#endif //BUFFER_H