
#include "../inc/buffer.h"

// __device__ inline unsigned int getWriteLoc(unsigned int* bufTail){
//     return atomicAdd(bufTail, 1);
// }
// __device__ unsigned int ldg (const unsigned int * p)
// {
//     unsigned int out;
//     asm volatile("ld.global.cg.s32 %0, [%1];" : "=r"(out) : "l"(p));
//     return out;
// }

__device__ void writeToBuffer(unsigned int* shBuffer,    unsigned int* glBuffer, unsigned int loc, unsigned int v){
    assert(loc < GLBUFFER_SIZE + MAX_NV);
    if(loc < MAX_NV)
        shBuffer[loc] = v;
    else
        glBuffer[loc-MAX_NV] = v;
}


__device__ unsigned int readFromBuffer(unsigned int* shBuffer,   unsigned int* glBuffer, unsigned int loc){
    assert(loc < GLBUFFER_SIZE + MAX_NV);
    return ( loc < MAX_NV ) ? shBuffer[loc] : glBuffer[loc-MAX_NV]; 
}



