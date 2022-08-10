
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
    
    if(write_to_disk){
        cout<<"Writing degrees to disk started... "<<endl;
        write_kcore_to_disk(data_graph.degrees, data_graph.V, "degrees.txt");
        cout<<"Writing degrees to disk completed... "<<endl;
    }
    
    cudaEventCreate(&event_start);
    cudaEventCreate(&event_stop);

    cout<<"start copying graph to gpu..."<<endl;
    malloc_graph_gpu_memory(data_graph, data_pointers);
    cout<<"end copying graph to gpu..."<<endl;

    unsigned int level = 0;
    unsigned int *global_count;
    volatile unsigned int* counter;
    cudaMallocManaged(&global_count,sizeof(unsigned int));
    cudaMallocManaged(&counter,sizeof(unsigned int));

    cudaMemset(global_count,0,sizeof(unsigned int));

    cudaEventRecord(event_start);

    size_t limit = 0;
    cudaDeviceGetLimit(&limit, cudaLimitMallocHeapSize);

    cout<<"default limit is: "<<limit<<endl;

    limit = 1024*1024*1024ULL;
    chkerr(cudaDeviceSetLimit(cudaLimitMallocHeapSize, limit));

    cudaDeviceGetLimit(&limit, cudaLimitMallocHeapSize);

    cout<<"new limit is: "<<limit<<endl;


	cout<<"Entering in while"<<endl;
	while(global_count[0] < data_graph.V && level<100){
        counter = 0;
        PKC<<<BLK_NUMS, BLK_DIM>>>(data_pointers, global_count, level, data_graph.V, counter);
        // test<<<BLK_NUMS, BLK_DIM>>>(data_pointers.degrees);
        chkerr(cudaDeviceSynchronize());
        cout<<"*********Completed level: "<<level<<", global_count: "<<global_count[0]<<" *********"<<endl;
        level += 1;
    }

	get_results_from_gpu(data_graph, data_pointers);

    cudaEventRecord(event_stop);
    cudaEventSynchronize(event_stop);


    float time_milli_sec = 0;
    cudaEventElapsedTime(&time_milli_sec, event_start, event_stop);
    cout<<"Elapsed Time: "<<time_milli_sec<<endl;

    
    if(write_to_disk){
        cout<<"Writing kcore to disk started... "<<endl;
        write_kcore_to_disk(data_graph.degrees, data_graph.V,data_file + "k-core.txt");
        cout<<"Writing kcore to disk completed... "<<endl;
    }

}

void find_kcore_CPU(string data_file,bool write_to_disk){
    cout<<"start loading graph file from disk to memory..."<<endl;    
    Graph data_graph(data_file);
    cout<<"graph loading complete..."<<endl;

    unsigned int level = 0;
    unsigned int count = 0;

    while(count<data_graph.V){

        for(int i=0; i<data_graph.V; i++){
            if (data_graph.degrees[i] == level){
                count++;
                unsigned int start = data_graph.neighbors_offset[i];
                unsigned int end = data_graph.neighbors_offset[i+1];

                for(unsigned int j = start; j<end; j++){
                    
                    unsigned int u = data_graph.neighbors[j];

                    if(data_graph.degrees[u] > level){
                        (data_graph.degrees[u])--;                        
                    }

                }
               
            }
        }    
        
        level++;
    }

}
