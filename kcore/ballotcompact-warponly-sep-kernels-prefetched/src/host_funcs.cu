
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
    
    // if(write_to_disk){
    //     cout<<"Writing degrees to disk started... "<<endl;
    //     write_kcore_to_disk(data_graph.degrees, data_graph.V, "degrees.txt");
    //     cout<<"Writing degrees to disk completed... "<<endl;
    // }
    
    cudaEventCreate(&event_start);
    cudaEventCreate(&event_stop);

    cout<<"start copying graph to gpu..."<<endl;
    malloc_graph_gpu_memory(data_graph, data_pointers);
    cout<<"end copying graph to gpu..."<<endl;

    unsigned int level = 0;
    unsigned int *global_count  = NULL;
    unsigned int* blockCounter  = NULL;
    unsigned int* glBuffers     = NULL;
    unsigned int* bufTails       = NULL;
    unsigned int count = 0;

    cudaMallocManaged(&global_count,sizeof(unsigned int));
    cudaMallocManaged(&blockCounter,sizeof(unsigned int));
    cudaMallocManaged(&bufTails,sizeof(unsigned int)*BLK_NUMS);
    

    cudaMemset(global_count,0,sizeof(unsigned int));

    cudaEventRecord(event_start);

    size_t limit = 0;
    cudaDeviceGetLimit(&limit, cudaLimitMallocHeapSize);

    cout<<"default limit is: "<<limit<<endl;

    limit = 10*1024*1024*1024ULL;
    chkerr(cudaDeviceSetLimit(cudaLimitMallocHeapSize, limit));
    limit = 0;
    cudaDeviceGetLimit(&limit, cudaLimitMallocHeapSize);
    chkerr(cudaMalloc(&glBuffers,sizeof(unsigned int)*BLK_NUMS*GLBUFFER_SIZE));

    cout<<"new limit is: "<<limit<<endl;
    auto start = chrono::steady_clock::now();
	cout<<"Entering in while"<<endl;
	while(count < data_graph.V){
        selectNodesAtLevel<<<BLK_NUMS, BLK_DIM>>>(data_pointers.degrees, bufTails, level, data_graph.V, glBuffers);
  
        PKC<<<BLK_NUMS, BLK_DIM>>>(data_pointers, global_count, level, data_graph.V, bufTails, glBuffers);
        // test<<<BLK_NUMS, BLK_DIM>>>(data_pointers.degrees);
        // chkerr(cudaDeviceSynchronize());
        chkerr(cudaMemcpy(&count, global_count, sizeof(unsigned int), cudaMemcpyDeviceToHost));

        cout<<"*********Completed level: "<<level<<", global_count: "<<count<<" *********"<<endl;
        level += 1;
    }

	get_results_from_gpu(data_graph, data_pointers);


    auto end = chrono::steady_clock::now();
    cout << "Elapsed Time: "
    << chrono::duration_cast<chrono::milliseconds>(end - start).count() << endl;

    free_graph_gpu_memory(data_pointers);
    cudaEventRecord(event_stop);
    cudaEventSynchronize(event_stop);
    cudaFree(glBuffers);


    
    if(write_to_disk){
        cout<<"Writing kcore to disk started... "<<endl;
        write_kcore_to_disk(data_graph.degrees, data_graph.V, data_file);
        cout<<"Writing kcore to disk completed... "<<endl;
    }

}