
#include "../inc/device_funcs.h"
#include "stdio.h"
#include "buffer.cc"

__global__ void selectNodesAtLevel(unsigned int *degrees, unsigned int level, unsigned int V, 
                 unsigned int* bufTails, unsigned int* glBuffers){

    __shared__ unsigned int* glBuffer; 
    __shared__ unsigned int* bufTail; 
    
    if(THID == 0){
        bufTail = bufTails + blockIdx.x;
        glBuffer = glBuffers + blockIdx.x*GLBUFFER_SIZE;
    }
    __syncthreads();

    unsigned int global_threadIdx = blockIdx.x * blockDim.x + threadIdx.x; 
    for(unsigned int base = 0; base < V; base += N_THREADS){
        
        unsigned int v = base + global_threadIdx; 

        if(v >= V) continue;

        if(degrees[v] == level){
            unsigned int loc = atomicAdd(bufTail, 1);
            writeToBuffer(glBuffer, loc, v);
        }
    }
}




__global__ void processNodes(G_pointers d_p, int level, int V, 
                    unsigned int* bufTails, unsigned int* glBuffers, 
                    unsigned int *global_count){

    __shared__ unsigned int shBuffer[MAX_NV];
    __shared__ unsigned int bufTail;
    __shared__ unsigned int* glBuffer;
    __shared__ unsigned int base;
    __shared__ unsigned int initTail;
    __shared__ unsigned int starts[32];
    __shared__ unsigned int ends[32];
    __shared__ bool prefetched[32];
    __shared__ int npref;

    unsigned int warp_id = THID / 32;
    unsigned int lane_id = THID % 32;
    unsigned int regTail;
    unsigned int i, start, end;
    if(THID==0){
        bufTail = bufTails[blockIdx.x];
        initTail = bufTail;
        base = 0;
        npref = 0;
        glBuffer = glBuffers + blockIdx.x*GLBUFFER_SIZE; 
        assert(glBuffer!=NULL);
    }

    if(lane_id == 0)
        prefetched[warp_id] = false;

    
    __syncthreads();

    // if(THID == 0 && level == 1)
    //     printf("%d ", bufTail);

    if(warp_id > 0)
        if(warp_id-1<bufTail){
            unsigned int v = readFromBuffer(shBuffer, glBuffer, initTail, warp_id-1);
            starts[warp_id] = d_p.neighbors_offset[v];
            ends[warp_id] = d_p.neighbors_offset[v+1];
        }
    if(THID==0){
        npref = min(WARPS_EACH_BLK-1, bufTail-base);
    }

    // if(THID == 0){
    //     base += WARPS_EACH_BLK-1;
    //     if(bufTail < base )
    //         base = bufTail;
    // }

    // bufTail is being incrmented within the loop, 
    // warps should process all the nodes added during the execution of loop
    
    // for(unsigned int i = warp_id; i<bufTail ; i +=warps_each_block ){
    // this for loop is a wrong choice, as many threads will exit from the loop checking the condition
    while(true){
        __syncthreads(); //syncthreads must be executed by all the threads
        if(warp_id < npref){
            start = starts[warp_id];
            end = ends[warp_id];
        }
        //todo check this condition
        if(base == bufTail) break; // all the threads will evaluate to true at same iteration
        i = base + warp_id - 1;
        regTail = bufTail;
        __syncthreads();


        if(warp_id == 0){
            if(lane_id == 0){
                // update base for next iteration
                base += npref;
            } 
            __syncwarp(); // so that other lanes can see updated base value
            if(lane_id > 0){
                int j = base + lane_id - 1;
                npref = min(WARPS_EACH_BLK-1, regTail-base);
                if(j < regTail){
                    unsigned int v = readFromBuffer(shBuffer, glBuffer, initTail, j);
                    starts[lane_id] = d_p.neighbors_offset[v];
                    ends[lane_id] = d_p.neighbors_offset[v+1];
                }
            }
            continue;
        }

        if(i >= regTail) continue;

        while(true){
            __syncwarp();

            if(start >= end) break;

            unsigned int j = start + lane_id;
            start += WARP_SIZE;
            if(j >= end) continue;

            unsigned int u = d_p.neighbors[j];
            if(ldg(d_p.degrees+u) > level){
                
                unsigned int a = atomicSub(d_p.degrees+u, 1);
            
                if(a == level+1){
                    unsigned int loc = atomicAdd(&bufTail, 1);
                    writeToBuffer(shBuffer, glBuffer, initTail, loc, u);
                }

                if(a <= level){
                    // node degree became less than the level after decrementing... 
                    atomicAdd(d_p.degrees+u, 1);
                }
            }
        }

    }

    if(THID == 0 && bufTail>0){
        atomicAdd(global_count, bufTail); // atomic since contention among blocks
    }
}
