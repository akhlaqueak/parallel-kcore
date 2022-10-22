
#include "../inc/device_funcs.h"
#include "stdio.h"
#include "buffer.cc"
#include "kcore.cc"
#include "util.cc"

__device__ inline void writeToTemp(unsigned int* tempv, unsigned int* templ, 
                            unsigned int v, unsigned int l, unsigned int& len){
    unsigned int laneid = LANEID;
    if(laneid == 0){
        tempv[len] = v;
        templ[len] = l;
        len++;
    }
}

__device__ int initializeSubgraph(Subgraphs sg, unsigned int len, unsigned int v){
    unsigned int* vtail = sg.vtail;
    unsigned int* otail = sg.otail;
    unsigned int laneid = LANEID;
    unsigned int ot, vt=0;
    if(laneid==0){
        vt = atomicAdd(vtail, len);
        ot = atomicAdd(otail, 2);
        sg.offsets[ot] = vt;
        sg.offsets[ot+1] = vt+len; 
        // insert v in the subgraph
        
        sg.vertices[vt] = v;
        sg.labels[vt] = R;
        vt++; // as one element is written i.e. v
    }    
    vt = __shfl_sync(FULL, vt, 0);
    return vt;
}
__device__ int getSubgraphTemp(G_pointers dp, Subgraphs sg, unsigned int s, unsigned int q){
    unsigned int warpid=WARPID;
    // unsigned int laneid=LANEID;
    unsigned int st = sg.otail[s];
    unsigned int en = sg.otail[s+1];
    unsigned int qst = dp.neighbors_offset[q];
    unsigned int qen = dp.neighbors_offset[q+1];
    unsigned int v, l, len = 0;
    // spawned subgraph len = 1 + |N(q) intersect (RUPUX)|
    // spawned subgraph:
    // R = q U (N(q) intersect R), or even simply R = q U R
    // P = N(q) intersect P
    // X = N(q) intersect X
    unsigned int* tempv = sg.tempv + warpid*TEMPSIZE;
    unsigned int* templ = sg.templ + warpid*TEMPSIZE;
    // todo intersection could be changed to binary search, but it'll cause divergence. Let's see in the future if it can help improve performance
    for(unsigned int i=st; i<en; i++){
        v = sg.vertices[i];
        l = sg.labels[i];
        if(l==R){ // it's already in N(q), no need to intersect. 
            // First lane writes it to buffer
            writeToTemp(tempv, templ, v, l, len); // len is updated inside this function
            continue;   
        }
        if(searchAny(dp.neighbors, qst, qen, v)){
            writeToTemp(tempv, templ, v, l, len); // len is updated inside this function
        }
    }
    // len is the number of items stored on temp buffer, let's generate subgraphs by adding q as R
    // len is updated all the time in lane0. now broadcast to other lanes
    len = __shfl_sync(FULL, len, 0);
    return len;
}


__device__ void generateSubGraphs(G_pointers dp, Subgraphs sg, unsigned int s, unsigned int q){
    unsigned int laneid = LANEID;
    unsigned int warpid = WARPID;
    printf("L%d", s);
    unsigned int len = getSubgraphTemp(dp, sg, s, q);
    unsigned int vt = initializeSubgraph(sg, len, q); // allocates a subgraph by atomic operations, and puts v as well
    unsigned int* tempv = sg.tempv + warpid*TEMPSIZE;
    unsigned int* templ = sg.templ + warpid*TEMPSIZE;
    for(unsigned int i=laneid; i<len; i+=32, vt+=32){
        unsigned int v = tempv[i];
        char label = templ[i];
        sg.vertices[vt] = v;
        sg.labels[vt] = label;
        if(label == Q)
            sg.labels[vt] = v<q? X : P;
    }
}    
__device__ void generateSubGraphs(G_pointers dp, Subgraphs sg, 
        unsigned int v){
    unsigned int laneid = LANEID;        
    unsigned int start = dp.neighbors_offset[v];
    unsigned int end = dp.neighbors_offset[v+1];
    unsigned int len = end-start+1; // number of neighbors + v itself
    if(len==1) return; // there was no neighbor for this vertex... 
    unsigned int vt, u;
    vt = initializeSubgraph(sg, len, v); // allocates a subgraph by atomic operations, and puts v as well
    // printf("vt:%d ", vt);
    for(unsigned int j=start+laneid, k=vt+laneid;j<end; j+=32, k+=32){
        u = dp.neighbors[j];
        sg.vertices[k] = u;
        if(u < v){sg.labels[k] = X;}
        else {sg.labels[k] = P;}
    }    
}    

__device__ void expandClique(G_pointers dp, Subgraphs sg, unsigned int s,  unsigned int pivot){
    
    unsigned int st = sg.offsets[s];
    unsigned int en = sg.offsets[s+1];
    unsigned int pst = dp.neighbors_offset[pivot];
    unsigned int pen = dp.neighbors_offset[pivot+1];
    unsigned int v;
    // subgraph stored in (st, en)
    // N(pivot) are in (pst, pen)
    // find Q=P-N(pivot)
    // for every u in Q, generate a subgraph
    printf("*%u:%u:%u*", s, st, en);
    for(unsigned int i = st; i<en; i++){
        if(sg.labels[i]!=P) continue; // only need to search for P
        v = sg.vertices[i];
        if(!searchAny(dp.neighbors, pst, pen, v)) // v belongs to Q, so generate subgraph for it
            sg.labels[i] = Q;             // simply change their labels to Q, afterwards generate a subgraph for each such node
        // this is necessary as per Algo 4 of the TPDS paper
    }
    // now generate subgraphs for all v's which were put to Q
    for(unsigned int i=st;i<en;i++)
        if(sg.labels[i] == Q)
            generateSubGraphs(dp, sg, s, sg.vertices[i]);
}

__device__ bool examineClique(Subgraphs sg, unsigned int s){
    unsigned int st = sg.offsets[s];
    unsigned int en = sg.offsets[s+1];
    return !searchAnyPX(sg.labels, st, en);
}

__device__ bool crossed(Subgraphs sg, unsigned int s){
    unsigned int st = sg.offsets[s];
    unsigned int en = sg.offsets[s+1];
    return !searchAnyP(sg.labels, st, en);
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
        if(sg.labels[j]==R) continue; // pivot is selected from P, X
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
        otail = 0;
        ohead = 0;
        vtail = 0;
    }
    __syncthreads();

    // create subgraphs... 
    unsigned int v = base+BLKID*SUBG+warpid;
    if(v<dp.V){
        generateSubGraphs(dp, sg, v);
    }

    __syncthreads();

    if(THID==0 && BLKID==0)
        printf("%d-%d", ohead, otail);
    
    while(true){
        __syncthreads();
        if(ohead >= otail) break;
        s = ohead + warpid*2;
        ohead = min(otail, ohead+WARPS_EACH_BLK*2);
        __syncthreads();
        if(s>=otail) continue;
        // printf("%d ", s);
        if(examineClique(sg, s)){
            // todo report clique
            // ?? do we need to store R, or just increment a count
            // seemingly GPU-BK(TPDS) is only calculating number of cliques
            dp.total++;
            printf("total: %d", dp.total);
        }
        else if(!crossed(sg, s)){
            unsigned int pivot = selectPivot(dp, sg, s);
            // printf("p%d", pivot);
            expandClique(dp, sg, s, pivot);
        }
    }
}


