
#include "../inc/host_funcs.h"
#include "../inc/gpu_memory_allocation.h"

void find_kcore(string data_file,bool write_to_disk){
    cout<<"Loading Started"<<endl;    
    Graph data_graph(data_file);
    cout<<"Loading Done"<<endl;
    G_pointers data_pointers;


    cout<<"Device Copy Started"<<endl;
    malloc_graph_gpu_memory(data_graph, data_pointers);
    cout<<"Device Copy Done"<<endl;

    unsigned int level = 0;
    unsigned int count = 0;
    unsigned int* global_count  = NULL;
    unsigned int* bufTails  = NULL;
    unsigned int* glBuffers     = NULL;

    chkerr(cudaMalloc(&global_count, sizeof(unsigned int)));
    chkerr(cudaMalloc(&bufTails, sizeof(unsigned int)*BLK_NUMS));
    cudaMemset(global_count, 0, sizeof(unsigned int));
    chkerr(cudaMalloc(&glBuffers,sizeof(unsigned int)*BLK_NUMS*GLBUFFER_SIZE));
    
    
    
	cout<<"K-core Computation Started"<<endl;

    auto start = chrono::steady_clock::now();
    while(count < data_graph.V){
        cudaMemset(bufTails, 0, sizeof(unsigned int)*BLK_NUMS);

        selectNodesAtLevel<<<BLK_NUMS, BLK_DIM>>>(data_pointers.degrees, level, 
                        data_graph.V, bufTails, glBuffers);

        processNodes<<<BLK_NUMS, BLK_DIM>>>(data_pointers, level, data_graph.V, 
                        bufTails, glBuffers, global_count);

        chkerr(cudaMemcpy(&count, global_count, sizeof(unsigned int), cudaMemcpyDeviceToHost));    
        cout<<"*********Completed level: "<<level<<", global_count: "<<count<<" *********"<<endl;
        level++;
    }
	cout<<"K-core Computation Done"<<endl;

    auto end = chrono::steady_clock::now();
    
    
    cout << "Elapsed Time: "
    << chrono::duration_cast<chrono::milliseconds>(end - start).count() << endl;
    cout <<"MaxK: "<<level-1<<endl;
    
    
	get_results_from_gpu(data_graph, data_pointers);
    
    cudaFree(glBuffers);
    free_graph_gpu_memory(data_pointers);
    if(write_to_disk){
        cout<<"Writing kcore to disk started... "<<endl;
        write_kcore_to_disk(data_graph.degrees, data_graph.V, data_file);
        cout<<"Writing kcore to disk completed... "<<endl;
    }

}
