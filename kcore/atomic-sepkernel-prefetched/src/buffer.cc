
#include "../inc/buffer.h"

// __device__ inline unsigned int getWriteLoc(unsigned int* bufTail){
//     return atomicAdd(bufTail, 1);
// }
__device__ unsigned int ldg (const unsigned int * p)
{
    unsigned int out;
    asm volatile("ld.global.cg.s32 %0, [%1];" : "=r"(out) : "l"(p));
    return out;
}

__device__ inline void writeToBuffer(unsigned int* glBuffer, unsigned int loc, unsigned int v){
    assert(loc < GLBUFFER_SIZE);
    glBuffer[loc] = v;
}

__device__ inline unsigned int readFromBuffer(unsigned int* glBuffer, unsigned int loc){
    assert(loc < GLBUFFER_SIZE);
    return glBuffer[loc];
}

__device__ void writeToBuffer(unsigned int* shBuffer, unsigned int* glBuffer, unsigned int initTail, unsigned int loc, unsigned int v){
    assert(loc < GLBUFFER_SIZE + MAX_NV);
    if((loc - initTail) < MAX_NV)
        shBuffer[loc-initTail] = v;
    else
        glBuffer[loc-MAX_NV] = v;
}


__device__ unsigned int readFromBuffer(unsigned int* shBuffer, unsigned int* glBuffer, unsigned int initTail, unsigned int loc){
    assert(loc < GLBUFFER_SIZE + MAX_NV);
    unsigned int v;

    if(loc < initTail) 
        v = glBuffer[loc];
    else if((loc - initTail) < MAX_NV) 
        v = shBuffer[loc-initTail];
    else 
        v = glBuffer[loc-MAX_NV];

    return v; 
}


