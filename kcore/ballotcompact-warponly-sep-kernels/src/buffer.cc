
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


__device__ inline unsigned int readFromBuffer( unsigned int* glBuffer, unsigned int loc){
    assert(loc < GLBUFFER_SIZE);
    return glBuffer[loc]; 
}



