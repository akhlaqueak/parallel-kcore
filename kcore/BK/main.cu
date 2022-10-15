#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>


#include "../inc/gpu_memory_allocation.h"
#include "../inc/device_funcs.h"

int find_kcore(Graph &g,bool write_to_disk){

    G_pointers dp;


    cout<<"Device Copy Started"<<endl;
    malloc_graph_gpu_memory(g, dp);
    cout<<"Device Copy Done"<<endl;
    unsigned int V = g.V;
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

    auto tick = chrono::steady_clock::now();
    while(count < g.V){
        cudaMemset(bufTails, 0, sizeof(unsigned int)*BLK_NUMS);

        selectNodesAtLevel<<<BLK_NUMS, BLK_DIM>>>(dp.degrees, level, 
                        g.V, bufTails, glBuffers);

        processNodes<<<BLK_NUMS, BLK_DIM>>>(dp, level, g.V, 
                        bufTails, glBuffers, global_count);

        chkerr(cudaMemcpy(&count, global_count, sizeof(unsigned int), cudaMemcpyDeviceToHost));    
        cout<<"*********Completed level: "<<level<<", global_count: "<<count<<" *********"<<endl;
        level++;
    }
	cout<<"K-core Computation Done"<<endl;
    cout<<"KMax: "<< level-1 <<endl;
    Graph gRec(g); // copy constructor overloaded... it allocates array for degree, neighbors... 
  
    
    unsigned int rec[g.V];
    chkerr(cudaMemcpy(&rec, dp.degOrder, sizeof(unsigned int)*V, cudaMemcpyDeviceToHost));    
    
    for(int i=0;i<g.V;i++)
        gRec.degrees[rec[i]] = g.degrees[i];

    gRec.neighbors_offset[0] = 0;
    std::partial_sum(gRec.degrees, gRec.degrees+V, gRec.neighbors_offset+1);

    for(int v=0;v<V;v++){
        unsigned int recv = rec[v];
        unsigned int start = gRec.neighbors_offset[recv];
        unsigned int end = gRec.neighbors_offset[recv+1];
        for (int j=g.neighbors_offset[v], k=start; j<g.neighbors_offset[v+1]; j++, k++){
            gRec.neighbors[k] = rec[g.neighbors[j]];
        }
        std::sort(gRec.neighbors+start, gRec.neighbors+end);

    }
    cout<<"Reordering Time: "<<chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now()-tick).count()<<endl;

    
    // cout << "Elapsed Time: "
    // << chrono::duration_cast<chrono::milliseconds>(end - start).count() << endl;
    // cout <<"MaxK: "<<level-1<<endl;
    
    
	// get_results_from_gpu(g, dp);
    
    cudaFree(glBuffers);
    free_graph_gpu_memory(dp);
    // if(write_to_disk){
    //     cout<<"Writing kcore to disk started... "<<endl;
    //     g.writeKCoreToDisk(data_file);
    //     cout<<"Writing kcore to disk completed... "<<endl;
    // }

    return chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - tick).count();

}


int main(int argc, char *argv[]){
    if (argc < 2) {
        cout<<"Please provide data file"<<endl;
        exit(-1);
    }
    std::string data_file = argv[1];
    bool write_to_disk = false;

    cout<<"Loading Started"<<endl;    
    Graph g(data_file);
    cout<<"Loading Done"<<endl;
    
    vector<int> et;
    for(int i=0;i<REP; i++){
        cout<<"Running iteration: "<<i+1<<endl;
        int t = find_kcore(g, write_to_disk);
        et.push_back(t);
    }
    cout << data_file << " Elapsed Time: ";

    for(auto t: et)
        cout<<t<<" ";
    cout<<(double)accumulate(et.begin(), et.end(), 0)/et.size();
    cout<<endl;
    return 0;
}
