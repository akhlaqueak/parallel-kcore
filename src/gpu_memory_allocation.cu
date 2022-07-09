/*
 * cuTS:  Scaling Subgraph Isomorphism on Distributed Multi-GPU Systems Using
 *        Trie Based Data Structure
 *
 * Copyright (C) 2021 APPL Laboratories (aravind_sr@outlook.com)
 *
 * This software is available under the MIT license, a copy of which can be
 * found in the file 'LICENSE' in the top-level directory.
 *
 * For further information contact:
 *   (1) Lizhi Xiang (lizhi.xiang@wsu.edu)
 *   (2) Aravind Sukumaran-Rajam (aravind_sr@outlook.com)
 *
 * The citation information is provided in the 'README' in the top-level
 * directory.
 */
#include "../inc/gpu_memory_allocation.h"
inline void chkerr(cudaError_t code)
{
    if (code != cudaSuccess)
    {
        std::cout<<cudaGetErrorString(code)<<std::endl;
        exit(-1);
    }
}
void malloc_graph_gpu_memory(Graph &g,G_pointers &p){
    chkerr(cudaMalloc(&(p.neighbors),g.neighbors_offset[g.V]*sizeof(unsigned int)));
    chkerr(cudaMemcpy(p.neighbors,g.neighbors,g.neighbors_offset[g.V]*sizeof(unsigned int),cudaMemcpyHostToDevice));
    chkerr(cudaMalloc(&(p.neighbors_offset),(g.V+1)*sizeof(unsigned int)));
    chkerr(cudaMemcpy(p.neighbors_offset,g.neighbors_offset,(g.V+1)*sizeof(unsigned int),cudaMemcpyHostToDevice));
    chkerr(cudaMalloc(&(p.degrees),(g.V)*sizeof(unsigned int)));
    chkerr(cudaMemcpy(p.degrees,g.degrees,(g.V)*sizeof(unsigned int),cudaMemcpyHostToDevice));
    p.V = g.V;
}

void get_results_from_gpu(Graph &g,G_pointers &p){
    chkerr(cudaMemcpy(g.degrees,p.degrees,(g.V)*sizeof(unsigned int),cudaMemcpyDeviceToHost));    
}
