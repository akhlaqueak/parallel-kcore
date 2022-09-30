#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>


#include "../inc/gpu_memory_allocation.h"
#include "../inc/device_funcs.h"

int find_kcore(Graph &data_graph,bool write_to_disk){

    G_pointers data_pointers;


    cout<<"Device Copy Started "<<data_graph.V<<data_graph.E<<endl;
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
        // cout<<"*********Completed level: "<<level<<", global_count: "<<count<<" *********"<<endl;
        level++;
    }
	cout<<"K-core Computation Done"<<endl;

    auto end = chrono::steady_clock::now();
    
    
    // cout << "Elapsed Time: "
    // << chrono::duration_cast<chrono::milliseconds>(end - start).count() << endl;
    // cout <<"MaxK: "<<level-1<<endl;
    
    
	// get_results_from_gpu(data_graph, data_pointers);
    
    cudaFree(glBuffers);
    free_graph_gpu_memory(data_pointers);
    // if(write_to_disk){
    //     cout<<"Writing kcore to disk started... "<<endl;
    //     data_graph.writeKCoreToDisk(data_file);
    //     cout<<"Writing kcore to disk completed... "<<endl;
    // }

    return chrono::duration_cast<chrono::milliseconds>(end - start).count();

}


int main(int argc, char *argv[]){
    if (argc < 2) {
        cout<<"Please provide data file"<<endl;
        exit(-1);
    }
    std::string data_file = argv[1];
    bool write_to_disk = false;

    cout<<"Loading Started"<<endl;    
    Graph data_graph(data_file);
    cout<<"Loading Done"<<endl;
    
    vector<int> et;
    for(int i=0;i<REP; i++){
        cout<<"Running iteration: "<<i+1<<endl;
        int t = find_kcore(data_graph, write_to_disk);
        et.push_back(t);
    }
    cout << data_file << " Elapsed Time: ";

    for(auto t: et)
        cout<<t<<" ";
    cout<<(double)accumulate(et.begin(), et.end(), 0)/et.size();
    cout<<endl;
    return 0;
}
