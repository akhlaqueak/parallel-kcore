#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>


#include "./inc/gpu_memory_allocation.h"
#include "./src/buffer.cc"
#include "./src/scans.cc"
#include "./src/ours.cc"
#include "./src/ours-shared.cc"
#include "./src/ours-prefetch.cc"


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
    unsigned int t;
    t = kcore(data_graph);
    cout<<"Ours: "<< t <<endl;

    t = kcoreSharedMem(data_graph);
    cout<<"Using shared memory buffer: "<< t <<endl;
    
    t = kcorePrefetch(data_graph);
    cout<<"Vertex prefetching: "<< t <<endl;
    
    // t = kcoreEfficientScan(data_graph);
    // cout<<"Compaction using Efficient scan: "<< t <<endl;
    
    // t = kcoreBallotScan(data_graph);
    // cout<<"Compaction using Ballot scan: "<< t <<endl;
    
    // t = kcoreBallotScanPrefetch(data_graph);
    // cout<<"Compaction using Ballot scan, vertex prefetching: "<< t <<endl;
    return 0;
}
