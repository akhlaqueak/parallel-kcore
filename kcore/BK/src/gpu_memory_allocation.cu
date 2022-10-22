
#include "../inc/gpu_memory_allocation.h"

void malloc_graph_gpu_memory(Graph &g,G_pointers &p){
    chkerr(cudaMalloc(&(p.neighbors),g.neighbors_offset[g.V]*sizeof(unsigned int)));
    chkerr(cudaMemcpy(p.neighbors,g.neighbors,g.neighbors_offset[g.V]*sizeof(unsigned int),cudaMemcpyHostToDevice));
    chkerr(cudaMalloc(&(p.neighbors_offset),(g.V+1)*sizeof(unsigned int)));
    chkerr(cudaMemcpy(p.neighbors_offset,g.neighbors_offset,(g.V+1)*sizeof(unsigned int),cudaMemcpyHostToDevice));
    chkerr(cudaMalloc(&(p.degrees),(g.V)*sizeof(unsigned int)));
    chkerr(cudaMemcpy(p.degrees,g.degrees,(g.V)*sizeof(unsigned int),cudaMemcpyHostToDevice));
    chkerr(cudaMalloc(&(p.degOrder),(g.V)*sizeof(unsigned int)));
    chkerr(cudaMalloc(&(p.total),sizeof(unsigned int)));
    cudaMemset(p.total, 0, sizeof(unsigned int));
    p.V = g.V;
    // std::cout<<"memory graph p = "<<p.neighbors[0]<<endl;
}
void recodedGraphCopy(Graph &g, G_pointers &p, Subgraphs** sg1, Subgraphs** sg2){
    chkerr(cudaMalloc(&(p.neighbors),g.neighbors_offset[g.V]*sizeof(unsigned int)));
    chkerr(cudaMemcpy(p.neighbors,g.neighbors,g.neighbors_offset[g.V]*sizeof(unsigned int),cudaMemcpyHostToDevice));
    chkerr(cudaMalloc(&(p.neighbors_offset),(g.V+1)*sizeof(unsigned int)));
    chkerr(cudaMemcpy(p.neighbors_offset,g.neighbors_offset,(g.V+1)*sizeof(unsigned int),cudaMemcpyHostToDevice));
    chkerr(cudaMalloc(&(p.degrees),(g.V)*sizeof(unsigned int)));
    chkerr(cudaMalloc(&(p.total),sizeof(unsigned int)));
    chkerr(cudaMemset(p.total, 0, sizeof(unsigned int)));
    chkerr(cudaMemcpy(p.degrees,g.degrees,(g.V)*sizeof(unsigned int),cudaMemcpyHostToDevice));
    p.V = g.V;
    chkerr(cudaMallocManaged(sg1, BLK_NUMS*sizeof(Subgraphs)));
    for(int i=0;i<BLK_NUMS; i++){
        chkerr(cudaMalloc(&(sg1[0][i].offsets), NSUBS*sizeof(unsigned int)));
        chkerr(cudaMalloc(&(sg1[0][i].vertices), NSUBS*1000*sizeof(unsigned int)));
        chkerr(cudaMalloc(&(sg1[0][i].labels), NSUBS*1000*sizeof(char)));
    }
    chkerr(cudaMallocManaged(sg2, BLK_NUMS*sizeof(Subgraphs)));
    for(int i=0;i<BLK_NUMS; i++){
        chkerr(cudaMalloc(&(sg2[0][i].offsets), NSUBS*sizeof(unsigned int)));
        chkerr(cudaMalloc(&(sg2[0][i].vertices), NSUBS*1000*sizeof(unsigned int)));
        chkerr(cudaMalloc(&(sg2[0][i].labels), NSUBS*1000*sizeof(char)));
    }
    for(int i=0;i<BLK_NUMS; i++){
        chkerr(cudaMalloc(&(sg1[0][i].tempv), TEMPSIZE*WARPS_EACH_BLK*sizeof(unsigned int)));
        chkerr(cudaMalloc(&(sg1[0][i].templ), TEMPSIZE*WARPS_EACH_BLK*sizeof(char)));
    }
}
void get_results_from_gpu(Graph &g,G_pointers &p){
    chkerr(cudaMemcpy(g.degrees,p.degrees,(g.V)*sizeof(unsigned int),cudaMemcpyDeviceToHost));    
}

void free_graph_gpu_memory(G_pointers &p){
    chkerr(cudaFree(p.neighbors));
    chkerr(cudaFree(p.neighbors_offset));
    chkerr(cudaFree(p.degrees));
}
