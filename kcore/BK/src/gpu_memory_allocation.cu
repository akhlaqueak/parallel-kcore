
#include "../inc/gpu_memory_allocation.h"

void malloc_graph_gpu_memory(Graph &g,G_pointers &p){
    chkerr(cudaMalloc(&(p.neighbors),g.neighbors_offset[g.V]*sizeof(unsigned int)));
    chkerr(cudaMemcpy(p.neighbors,g.neighbors,g.neighbors_offset[g.V]*sizeof(unsigned int),cudaMemcpyHostToDevice));
    chkerr(cudaMalloc(&(p.neighbors_offset),(g.V+1)*sizeof(unsigned int)));
    chkerr(cudaMemcpy(p.neighbors_offset,g.neighbors_offset,(g.V+1)*sizeof(unsigned int),cudaMemcpyHostToDevice));
    chkerr(cudaMalloc(&(p.degrees),(g.V)*sizeof(unsigned int)));
    chkerr(cudaMemcpy(p.degrees,g.degrees,(g.V)*sizeof(unsigned int),cudaMemcpyHostToDevice));
    chkerr(cudaMalloc(&(p.degOrder),(g.V)*sizeof(unsigned int)));
    p.V = g.V;
    // std::cout<<"memory graph p = "<<p.neighbors[0]<<endl;
}
void recodedGraphCopy(Graph &g, G_pointers &p, Subgraphs** sg){
    chkerr(cudaMalloc(&(p.neighbors),g.neighbors_offset[g.V]*sizeof(unsigned int)));
    chkerr(cudaMemcpy(p.neighbors,g.neighbors,g.neighbors_offset[g.V]*sizeof(unsigned int),cudaMemcpyHostToDevice));
    chkerr(cudaMalloc(&(p.neighbors_offset),(g.V+1)*sizeof(unsigned int)));
    chkerr(cudaMemcpy(p.neighbors_offset,g.neighbors_offset,(g.V+1)*sizeof(unsigned int),cudaMemcpyHostToDevice));
    chkerr(cudaMalloc(&(p.degrees),(g.V)*sizeof(unsigned int)));
    chkerr(cudaMemcpy(p.degrees,g.degrees,(g.V)*sizeof(unsigned int),cudaMemcpyHostToDevice));
    p.V = g.V;
    chkerr(cudaMalloc(sg, BLK_NUMS*sizeof(Subgraphs)));

    for(int i=0;i<BLK_NUMS; i++){
        chkerr(cudaMalloc(&(sg[i].offsets), NSUBS*sizeof(unsigned int)));
        chkerr(cudaMalloc(&(sg[i].vertices), NSUBS*1000*sizeof(unsigned int)));
        chkerr(cudaMalloc(&(sg[i].labels), NSUBS*1000*sizeof(char)));
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
