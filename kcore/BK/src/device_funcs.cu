
#include "../inc/device_funcs.h"
#include "stdio.h"
#include "buffer.cc"
#include "kcore.cc"
#include "util.cc"

__device__ int generateSubGraphs(G_pointers dp, Subgraphs sg, 
        unsigned int v){
    unsigned int laneid = LANEID;
    unsigned int* otail = sg.otail;
    unsigned int* vtail = sg.vtail;
    unsigned int start = dp.neighbors_offset[v];
    unsigned int end = dp.neighbors_offset[v+1];
    unsigned int len = end-start+1; // number of neighbors + v itself
    if(len==1) return 0; // there was no neighbor for this vertex... 
    unsigned int loc, u, ot;
    if(laneid == 0){
        loc = atomicAdd(vtail, len);
        ot = atomicAdd(otail, 2);
        sg.offsets[ot] = loc;
        sg.offsets[ot+1] = loc+len; 
        // insert v in the subgraph
        sg.vertices[loc] = v;
        sg.labels[loc] = R;
        loc++; // as one element is written already... 
    }
    loc = __shfl_sync(FULL, loc, 0);
    for(unsigned int j=start+laneid, k=loc+laneid;j<end; j+=32, k+=32){
        u = dp.neighbors[j];
        sg.vertices[k] = u;
        if(u < v){sg.labels[k] = X;}
        else {sg.labels[k] = P;}
    }
    return ot;
}

__device__ void fixR(Subgraphs sg, unsigned int ot, unsigned int st, unsigned int en){
    unsigned int sgst = sg.offsets[ot];
    unsigned int sgen = sg.offsets[ot+1];
    unsigned int laneid=LANEID;
    for(unsigned int i=sgst; i<sgen; i++){
        if(sg.labels[i]==R) continue; // this node already have status R
        unsigned int v = sg.vertices[i];
        for(unsigned int k=st+laneid; k<en; k+=32){            
            if (v == sg.vertices[k] && sg.labels[k]==R){
                sg.labels[i]=R;
                break;
            }
        }
    }
}


__device__ void expandClique(G_pointers dp, Subgraphs sg, unsigned int i,  unsigned int pivot){

    unsigned int st = sg.otail[i];
    unsigned int en = sg.otail[i+1];

    // unsigned int warpid=WARPID;
    // unsigned int laneid=LANEID;
    unsigned int v, pst, pen;
    pst = dp.neighbors_offset[pivot];
    pen = dp.neighbors_offset[pivot+1];
    // subgraph stored in (st, en)
    // N(pivot) are in (pst, pen)
    for(unsigned int i = st; i<en; i++){
        // entire warp is reading one vertex from subgraph, hence not adding laneid in i
        if(sg.labels[i]!=P) continue;
        v = sg.vertices[i]; // v in P
        

        if(!searchAtWarpAny(dp.neighbors, pst, pen, v)){ //this v is in P-N(pivot)
            // generate a subgraph for this sg.vertices[i]: done
            unsigned int ot = generateSubGraphs(dp, sg, v);
            // fux R in newly spawned graph, all such vertices which are R in old graph are R in spawned graph
            fixR(sg, ot, st, en);
        }
    }
}

__device__ bool examineClique(Subgraphs sg, unsigned int i){
    unsigned int st = sg.offsets[i];
    unsigned int en = sg.offsets[i+1];
    return searchAtWarpAll(sg.labels, st, en, R);
    // report = true;
    // for(; st<en; st+=32){
    //     j = st+laneid;
    //     if(j>=en) pred = true; // these are non-participating threads, hence getting value true
    //     else pred = sg.labels[j]==R;
    //     res = __ballot_sync(FULL, pred);
    //     if(res!=FULL){
    //         report = false;
    //         break;
    //     }
    // }
    // return report;
}



__device__ unsigned int selectPivot(G_pointers dp, Subgraphs sg, unsigned int i){
    unsigned int laneid=LANEID;
    unsigned int st=sg.offsets[i];
    unsigned int en=sg.offsets[i+1];
    unsigned int v;
    unsigned int st1, en1;
    unsigned int nmatched, max, pivot;
    bool pred;

    for(unsigned int j=st; j<en; j++){// entire warp is processing one element, hence laneid is not added...
        // it's not a divergence, entire warp will continue as result of below condition
        if(sg.labels[j]==R) continue; // pivot is selected from P and X
        v = sg.vertices[j];             // v: X or P
        // (st1, en1) are N(v)
        st1 = dp.neighbors_offset[v];
        en1 = dp.neighbors_offset[v+1];
        nmatched = 0, max = 0, pivot = v;
        for(unsigned int k=st; k<en; k+=32){
            unsigned int kl = k+laneid; // need to run all lanes, so that ballot function works well
            pred = false;
            // some of the lanes will diverge from this point, it may be improved in future.
            if(kl<en && sg.labels[kl]==P)  // only P nodes will be searched.
                pred = binarySearch(dp.neighbors+st1, en1-st1, sg.vertices[kl]); // P intersect N(v)
                // binary search can introduce divergence, we can also try with warp level linear search in future
            nmatched+=__popc(__ballot_sync(FULL, pred));
        }
        if(nmatched > max){
            max = nmatched;
            pivot = v;
        }
    }
    return pivot;
}


__global__ void BK(G_pointers dp, Subgraphs* subgs, unsigned int base){
    __shared__ Subgraphs sg;
    __shared__ unsigned int vtail;
    __shared__ unsigned int otail;
    __shared__ unsigned int ohead;
    // vtail: vertices tail, a subgraph vertices stored based on an atomic increment to it
    //          labels also use the same vtail
    // otail: offset tail, two consective values represent start and end of a subgraph.
    //          it's always atomically incremented by 2.

    unsigned int warpid = WARPID;
    unsigned int s;
    if(THID==0){
        sg = subgs[BLKID];
        sg.otail = &otail;
        sg.vtail = &vtail;
        sg.ohead = &ohead;
        sg.otail[0] = 0;
        sg.ohead[0] = 0;
        sg.vtail[0] = 0;
    }
    __syncthreads();

    // create subgraphs... 
    unsigned int v = base+BLKID*SUBG+warpid;
    if(v<dp.V){
        generateSubGraphs(dp, sg, v);
    }
    
    while(true){
        __syncthreads();
        if(ohead >= otail) break;
        s = ohead + warpid*2;
        ohead = min(otail, ohead+WARPS_EACH_BLK*2);
        __syncthreads();

        if(examineClique(sg, s)){
            // todo report clique
            // ?? do we need to store R, or just increment a count
            // seemingly GPU-BK(TPDS) is only calculating number of cliques
        }
        else{
            unsigned int pivot = selectPivot(dp, sg, s);
            expandClique(dp, sg, s, pivot);
        }
    }

    // if(THID==0 && BLKID==0)
    // for(int i=0;i<otail;i+=2){
    //     unsigned int st = sg.offsets[i];
    //     unsigned int en = sg.offsets[i+1];
    //     printf("%d-%d:", st, en);
    //     for(;st<en;st++){
    //         printf("%d%c ", sg.vertices[st], sg.labels[st]);
    //     }
    //     printf("\n");
    // }
}


