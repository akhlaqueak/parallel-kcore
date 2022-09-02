
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

 

    cout<<"start copying graph to gpu..."<<endl;
    malloc_graph_gpu_memory(data_graph, data_pointers);
    cout<<"end copying graph to gpu..."<<endl;

    unsigned int level = 0;
    unsigned int count = 0;
    unsigned int *global_count  = NULL;
    unsigned int* blockCounter  = NULL;
    unsigned int* glBuffers     = NULL;

    chkerr(cudaMalloc(&global_count, sizeof(unsigned int)));
    chkerr(cudaMalloc(&blockCounter, sizeof(unsigned int)));
    cudaMemset(global_count, 0, sizeof(unsigned int));
    
    
    
    size_t limit = 0;
    cudaDeviceGetLimit(&limit, cudaLimitMallocHeapSize);
    
    cout<<"default limit is: "<<limit<<endl;
    
    limit = 1024*1024*1024ULL;
    chkerr(cudaDeviceSetLimit(cudaLimitMallocHeapSize, limit));
    limit = 0;
    cudaDeviceGetLimit(&limit, cudaLimitMallocHeapSize);
    chkerr(cudaMalloc(&glBuffers,sizeof(unsigned int)*BLK_NUMS*GLBUFFER_SIZE));
    
    cout<<"new limit is: "<<limit<<endl;
    
    auto start = chrono::steady_clock::now();
    
	cout<<"Entering in while"<<endl;
	while(count < data_graph.V){
        cudaMemset(blockCounter,0,sizeof(unsigned int));        
        PKC<<<BLK_NUMS, BLK_DIM>>>(data_pointers, global_count, level, data_graph.V, blockCounter, glBuffers);
        // test<<<BLK_NUMS, BLK_DIM>>>(data_pointers.degrees);
        // chkerr(cudaDeviceSynchronize());
        chkerr(cudaMemcpy(&count, global_count, sizeof(unsigned int), cudaMemcpyDeviceToHost));    
        
        cout<<"*********Completed level: "<<level<<", global_count: "<<count<<" *********"<<endl;
        level += 1;
    }

	get_results_from_gpu(data_graph, data_pointers);


    cudaFree(glBuffers);



    auto end = chrono::steady_clock::now();
    cout << "Elapsed Time: "
    << chrono::duration_cast<chrono::milliseconds>(end - start).count() << endl;
    
    if(write_to_disk){
        cout<<"Writing kcore to disk started... "<<endl;
        write_kcore_to_disk(data_graph.degrees, data_graph.V, data_file);
        cout<<"Writing kcore to disk completed... "<<endl;
    }

}
