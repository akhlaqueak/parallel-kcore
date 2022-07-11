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
#include "../inc/host_funcs.h"
#include "../inc/gpu_memory_allocation.h"
inline void chkerr(cudaError_t code)
{
    if (code != cudaSuccess)
    {
        std::cout<<cudaGetErrorString(code)<<std::endl;
        exit(-1);
    }
}
void copy_graph_to_gpu(Graph data_graph, G_pointers &data_pointers){
    malloc_graph_gpu_memory(data_graph,data_pointers);
}
void find_kcore(string data_file,bool write_to_disk){
    cout<<"start loading graph file from disk to memory..."<<endl;
    
    Graph data_graph(data_file);

    cout<<"graph loading complete..."<<endl;
    G_pointers data_pointers;

    cudaEvent_t event_start;
    cudaEvent_t event_stop;
    cudaEventCreate(&event_start);
    cudaEventCreate(&event_stop);

    cout<<"start copying graph to gpu..."<<endl;
    copy_graph_to_gpu(data_graph, data_pointers);
    cout<<"end copying graph to gpu..."<<endl;

    unsigned int level = 0;
    unsigned int *global_count;
    cudaMallocManaged(&global_count,sizeof(unsigned int));

    cudaMemset(global_count,0,sizeof(unsigned int));

    cudaEventRecord(event_start);
	cout<<"Entering in while"<<endl;
	// while(global_count[0] < data_graph.V){
        for(int i=0;i<2;i++){
	cout<<"level: "<<level<<", global_count: "<<global_count[0]<<endl;
        PKC<<<BLK_NUMS, BLK_DIM>>>(data_pointers, global_count, level);
        level += 1;
        // chkerr(cudaDeviceSynchronize());
    }

	// get_results_from_gpu(data_graph, data_pointers);

    // cudaEventRecord(event_stop);
    // cudaEventSynchronize(event_stop);


    // float time_milli_sec = 0;
    // cudaEventElapsedTime(&time_milli_sec, event_start, event_stop);
    // cout<<"Elapsed Time: "<<time_milli_sec<<endl;

    
    // if(write_to_disk){
    //     cout<<"Writing kcore to disk started... "<<endl;
    //     write_kcore_to_disk(data_graph.degrees, data_graph.V);
    //     cout<<"Writing kcore to disk completed... "<<endl;
    // }

}
