
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



__device__ bool allocationRequired( unsigned int* glBuffer, unsigned int loc){
    return (glBuffer == NULL && // global buffer is not allocated before
            loc >= MAX_NV); //shBuffer is exhausted
}
__device__ void allocateMemory( unsigned int** glBufferPtr){
        glBufferPtr[0] = ( unsigned int*) malloc(sizeof(unsigned int) * GLBUFFER_SIZE);
        // printf("allocated %d %d\n", blockIdx.x, THID);
        assert(glBufferPtr[0]!=NULL);        
}

__device__ void allocateMemoryMutex( unsigned int** glBufferPtr, volatile unsigned int* lock){
    if(atomicExch((unsigned int*)lock, 1) == 0){        
        // printf("mutex %d %d\n", blockIdx.x, THID);
        allocateMemory(glBufferPtr);
        lock[0] = 2; // not necessary to do it atomically, since it's the only thread in critical section
        __threadfence_block(); // it ensures the writes done to shared/global mem
    }
    while(lock[0]!=2);
}    