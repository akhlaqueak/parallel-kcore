
#include "../inc/device_funcs.h"
#include "stdio.h"
#include "buffer.cc"
#include "kcore.cc"

__device__ void generateSubGraphs(G_pointers dp, Subgraphs sg, 
        unsigned int v, unsigned int* otail, unsigned int* vtail){
    unsigned int laneid = LANEID;
    unsigned int warpid = WARPID;
    unsigned int start = dp.neighbors_offset[v];
    unsigned int end = dp.neighbors_offset[v+1];
    unsigned int len = end-start+1; // number of neighbors + v itself
    if(len==1) return; // there was no neighbor this vertex... 
    unsigned int loc, u;
    if(laneid == 0){
        loc = atomicAdd(vtail, len);
        unsigned int st = atomicAdd(otail, 2);
        sg.offsets[st] = loc;
        sg.offsets[st+1] = loc+len; 

        sg.vertices[loc] = v;
        sg.labels[loc] = R;
        printf("%d=*", sg.vertices[loc]);
        for(int i=start; i<end; i++){
            printf("%d,", dp.neighbors[i]);
            printf("\n");
        }
        loc++; // as one element is written already... 
    }
    loc = __shfl_sync(FULL, loc, 0);
    for(unsigned int j=start+laneid, k=loc+laneid;j<end; j+=32, k+=32){
        u = dp.neighbors[j];
        sg.vertices[k] = u;
        if(u < v){sg.labels[k] = X;}
        else {sg.labels[k] = P;}
    }
}


__global__ void BK(G_pointers dp, Subgraphs* subgs, unsigned int base){
    __shared__ Subgraphs sg;
    __shared__ unsigned int vtail;
    __shared__ unsigned int otail;
    // vtail: vertices tail, a subgraph vertices stored based on an atomic increment to it
    //          labels also use the same vtail
    // otail: offset tail, two consective values represent start and end of a subgraph.
    //          it's always atomically incremented by 2.

    unsigned int warpid = WARPID;
    unsigned int laneid = LANEID;
    if(THID==0){
        sg = subgs[BLKID];
        vtail = 0;
        otail = 0;
    }
    __syncthreads();

    // create subgraphs... 
    unsigned int v = base+BLKID*SUBG+warpid;
    if(v<dp.V){
        generateSubGraphs(dp, sg, v, &otail, &vtail);
    }
    __syncthreads();
    if(THID==0 && BLKID==0)
    for(int i=0;i<otail;i+=2){
        unsigned int st = sg.offsets[i];
        unsigned int en = sg.offsets[i+1];
        printf("%d-%d:", st, en);
        for(;st<en;st++){
            printf("%d%c ", sg.vertices[st], sg.labels[st]);
        }
        printf("\n");
    }
}


