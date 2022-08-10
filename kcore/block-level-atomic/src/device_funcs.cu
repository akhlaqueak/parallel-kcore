
#include "../inc/device_funcs.h"
#include "stdio.h"

__device__ unsigned int getWriteLoc(unsigned int** glBuffer, unsigned int* bufTail){
    unsigned int loc = atomicAdd(bufTail, 1);
    assert(loc < GLBUFFER_SIZE + MAX_NV);

    // if(loc == MAX_NV){ // checking equal so that only one thread in a warp should allocate glBuffer
    //     glBuffer[0] = (unsigned int*) malloc(sizeof(unsigned int) * GLBUFFER_SIZE); 
    //     assert(glBuffer[0] != NULL); 
    // }
    return loc;
}

__device__ void writeToBuffer(unsigned int* shBuffer,  unsigned int** glBuffer, unsigned int loc, unsigned int v){
    // todo: make it single pointer, glBuffer
    assert(loc < GLBUFFER_SIZE + MAX_NV);

    if(loc < MAX_NV){
        shBuffer[loc] = v;
    }
    else{
        assert(glBuffer[0]!=NULL);
        glBuffer[0][loc-MAX_NV] = v; 
    }
}

__device__ void selectNodesAtLevel(unsigned int *degrees, unsigned int V, unsigned int* shBuffer, unsigned int** glBuffer, unsigned int* bufTail, unsigned int level){
    unsigned int global_threadIdx = blockIdx.x * blockDim.x + threadIdx.x; 
    for(unsigned int i=global_threadIdx; i<V; i+= N_THREADS){
        if(degrees[i] == level){
            unsigned int loc = getWriteLoc(glBuffer, bufTail);
            writeToBuffer(shBuffer, glBuffer, loc, i);
        }
    }
    __syncthreads();
}

__device__ unsigned int readFromBuffer(unsigned int* shBuffer, unsigned int** glBuffer, unsigned int loc){
    assert(loc < MAX_NV + GLBUFFER_SIZE);
    return ( loc < MAX_NV ) ? shBuffer[loc] : glBuffer[0][loc-MAX_NV]; 
}

__global__ void PKC(G_pointers d_p, unsigned int *global_count, int level, int V){


    __shared__ unsigned int shBuffer[MAX_NV];
    __shared__ unsigned int bufTail;
    __shared__ unsigned int* glBuffer;
    __shared__ unsigned int base;
    unsigned int warp_id = THID / 32;
    unsigned int lane_id = THID % 32;
    unsigned int i;

    if(THID == 0){
        bufTail = 0;
        glBuffer = NULL;
        base = 0;
        glBuffer = (unsigned int*) malloc(sizeof(unsigned int) * GLBUFFER_SIZE); 
    }

    __syncthreads();

    selectNodesAtLevel(d_p.degrees, V, shBuffer, &glBuffer, &bufTail, level);

    

    // bufTail is being incrmented within the loop, 
    // warps should process all the nodes added during the execution of loop
    // for that purpose e_processes is introduced, is incremented whenever a warp takes a job. 
    
    
    // for(unsigned int i = warp_id; i<bufTail ; i = warp_id + base){
    // this for loop is a wrong choice, as many threads will exit from the loop checking the condition
    while(true){
        __syncthreads(); //syncthreads must be executed by all the threads, so can't put after break or continue...

        if(base == bufTail) break;

        i = base + warp_id;
        
        if(THID == 0){
            base += WARPS_EACH_BLK;
            if(bufTail < base )
                base = bufTail;
        }
        __syncthreads();
        if(i >= bufTail) continue; // this warp won't have to do anything     
        
        
        unsigned int v, start, end;

        v = readFromBuffer(shBuffer, &glBuffer, i);
        start = d_p.neighbors_offset[v];
        end = d_p.neighbors_offset[v+1];


        for(int j = start + lane_id; j<end ; j+=32){
            unsigned int u = d_p.neighbors[j];
            if(d_p.degrees[u] > level){
                unsigned int a = atomicSub(d_p.degrees+u, 1);
            
                if(a == level+1){
                    unsigned int loc = getWriteLoc(&glBuffer, &bufTail);
                    writeToBuffer(shBuffer, &glBuffer, loc, u);
                }

                if(a <= level){
                    // node degree became less than the level after decrementing... 
                    atomicAdd(d_p.degrees+u, 1);
                }
            }
        }

    }


    if(THID == 0 ){
        atomicAdd(global_count, bufTail); // atomic since contention among blocks
        if(glBuffer!=NULL) free(glBuffer);
    }

}


