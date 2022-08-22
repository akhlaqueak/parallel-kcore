#ifndef BUFFER_H
#define BUFFER_H

#include "./common.h"


#include "../inc/buffer.h"



__device__ void writeToBuffer(unsigned int* shBuffer, Node* tail, unsigned int loc, unsigned int v);

__device__ void advanceNode(Node** head);

__device__ unsigned int readFromBuffer(unsigned int* shBuffer, Node* head, unsigned int loc);

__device__ bool allocationRequired( Node* tail, unsigned int loc);

__device__ void allocateMemory(Node** tail, Node** head);

__device__ void allocateMemoryMutex(Node** tail, Node** head, volatile unsigned int* lock);

#endif //BUFFER_H