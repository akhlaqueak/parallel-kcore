
#include "../inc/host_funcs.h"
#include "../inc/gpu_memory_allocation.h"
#define REPORTTIME cout<<"Elapsed Time: "<<chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now()-tick).count()<<endl;
    
inline void chkerr(cudaError_t code)
{
    if (code != cudaSuccess)
    {
        std::cout<<cudaGetErrorString(code)<<std::endl;
        exit(-1);
    }
}
void copy_graph_to_gpu(Graph g, G_pointers &data_pointers){
    malloc_graph_gpu_memory(g,data_pointers);
}
void find_kcore(string data_file,bool write_to_disk){
    cout<<"start loading graph file from disk to memory..."<<endl;
    
    Graph g(data_file);
    unsigned int V = g.V;

    cout<<"graph loading complete..."<<endl;
    G_pointers data_pointers;


    cout<<"start copying graph to gpu..."<<endl;
    malloc_graph_gpu_memory(g, data_pointers);
    cout<<"end copying graph to gpu..."<<endl;

    unsigned int level = 0;
    unsigned int count = 0;
    unsigned int* global_count  = NULL;
    unsigned int* bufTails  = NULL;
    unsigned int* glBuffers     = NULL;

    chkerr(cudaMalloc(&global_count, sizeof(unsigned int)));
    chkerr(cudaMalloc(&bufTails, sizeof(unsigned int)*BLK_NUMS));
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
    
    
	cout<<"Entering in while"<<endl;

    auto tick = chrono::steady_clock::now();
    while(count < g.V){
        cudaMemset(bufTails, 0, sizeof(unsigned int)*BLK_NUMS);
        // chkerr(cudaDeviceSynchronize());
        selectNodesAtLevel<<<BLK_NUMS, BLK_DIM>>>(data_pointers.degrees, level, g.V, bufTails, glBuffers);

        processNodes<<<BLK_NUMS, BLK_DIM>>>(data_pointers, level, g.V, bufTails, glBuffers, global_count);
        // test<<<BLK_NUMS, BLK_DIM>>>(data_pointers.degrees);
        chkerr(cudaMemcpy(&count, global_count, sizeof(unsigned int), cudaMemcpyDeviceToHost));    
        
        cout<<"*********Completed level: "<<level<<", global_count: "<<count<<" *********"<<endl;
        level++;
    }
    REPORTTIME;
    cout <<"MaxK: "<<level-1<<endl;
    tick = chrono::steady_clock::now();


    // recode the graph as per degeneracy order
    Graph gRec(g); // copy constructor overloaded... it allocates array for degree, neighbors... 
  
    
    unsigned int rec[g.V];
    chkerr(cudaMemcpy(&rec, data_pointers.degOrder, sizeof(unsigned int)*V, cudaMemcpyDeviceToHost));    
    
    for(int i=0;i<g.V;i++)
        gRec.degrees[rec[i]] = g.degrees[i];

    gRec.neighbors_offset[0] = 0;
    std::partial_sum(gRec.neighbors_offset, gRec.neighbors_offset+V, gRec.neighbors_offset+1);

    for(int v=0;v<V;v++){
        unsigned int recv = rec[v];
        cout<<v<<" -> "<<recv<<endl;
        for (int j=g.neighbors_offset[v], k=gRec.neighbors_offset[recv]; j<g.neighbors_offset[v+1]; j++, k++){
            gRec.neighbors[k] = rec[g.neighbors[j]];
            cout<<gRec.neighbors[k]<<" ";
        }
        std::sort(gRec.neighbors[gRec.neighbors_offset[recv]], gRec.neighbors[gRec.neighbors_offset[recv+1]]);
        cout<<endl;
    }
    REPORTTIME;
    cudaFree(glBuffers);
    free_graph_gpu_memory(data_pointers);
    if(write_to_disk){
        cout<<"Writing kcore to disk started... "<<endl;
        write_kcore_to_disk(gRec.degrees, gRec.V, data_file);
        cout<<"Writing kcore to disk completed... "<<endl;
    }

}
