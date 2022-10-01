#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>


#include "./inc/gpu_memory_allocation.h"
#include "./src/buffer.cc"
#include "./src/scans.cc"
#include "./src/ours.cc"
#include "./src/ours-shared.cc"
#include "./src/ours-prefetch.cc"
#include "./src/efficient.cc"
#include "./src/ballot-prefetch.cc"
#include "./src/ballot.cc"


int main(int argc, char *argv[]){
    if (argc < 2) {
        cout<<"Please provide data file"<<endl;
        exit(-1);
    }
    std::string data_file = argv[1];

    cout<<"Graph loading Started... ";    
    Graph data_graph(data_file);
    cout<<"Done"<<endl;
    unsigned int t;

    cout<<"V: "<< data_graph.V<<endl;
    cout<<"E: "<< data_graph.E<<endl;

    cout<<"Computing ours... "<<endl;
    t = kcore(data_graph);
    cout<<"Kmax: "<<data_graph.Kmax<<endl;
    cout<<"Done: "<< t << "ms" << endl;

    cout<<"Computing ours algo with using shared memory buffer... "<<endl;
    t = kcoreSharedMem(data_graph);
    cout<<"Done: "<< t << "ms" << endl;
    
    cout<<"Computing ours algo with vertex prefetching... "<<endl;
    t = kcorePrefetch(data_graph);
    cout<<"Done: "<< t << "ms" << endl;

    cout<<"Computing using Efficient scan: "<<endl;
    t = kcoreEfficientScan(data_graph);
    cout<<"Done: "<< t << "ms" << endl;
    
    cout<<"Computing using Ballot scan: " <<endl;
    t = kcoreBallotScan(data_graph);
    cout<<"Done: "<< t << "ms" << endl;
    
    cout<<"Computing Ballot scan, vertex prefetching: "<<endl;
    t = kcoreBallotScanPrefetch(data_graph);
    cout<<"Done: "<< t << "ms" << endl;
    
    return 0;
}
