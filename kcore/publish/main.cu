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
#include "./src/efficient-prefetch.cc"
#include "./src/ballot-shared.cc"
#include "./src/efficient-shared.cc"


int main(int argc, char *argv[]){
    if (argc < 2) {
        cout<<"Please provide data file"<<endl;
        exit(-1);
    }
    std::string data_file = argv[1];

    cout<<"Graph loading Started... "<<endl;    
    Graph data_graph(data_file);
    unsigned int t;

    cout<<"V: "<< data_graph.V<<endl;
    cout<<"E: "<< data_graph.E<<endl;

    // cout<<"Computing ours... ";
    t = kcore(data_graph);
    cout<<"Kmax: "<<data_graph.kmax<<endl;
    cout<<"Our algo Done: "<< t << "ms" << endl<< endl;

    cout<<"Computing Shared Memory + Ours... ";
    t = kcoreSharedMem(data_graph);
    cout<<"Done: "<< t << "ms" << endl<< endl;
    
    cout<<"Computing Vertex Prefetching + Ours ... ";
    t = kcorePrefetch(data_graph);
    cout<<"Done: "<< t << "ms" << endl<< endl;

    cout<<"Computing Efficient scan: ";
    t = kcoreEfficientScan(data_graph);
    cout<<"Done: "<< t << "ms" << endl<< endl;
    
    cout<<"Computing Ballot scan: " ;
    t = kcoreBallotScan(data_graph);
    cout<<"Done: "<< t << "ms" << endl<< endl;
    
    cout<<"Computing Ballot scan + Vertex Prefetching: ";
    t = kcoreBallotScanPrefetch(data_graph);
    cout<<"Done: "<< t << "ms" << endl<< endl;

    cout<<"Computing Efficient Scan, Vertex Prefetching: ";
    t = kcoreEfficientScanPrefetch(data_graph);
    cout<<"Done: "<< t << "ms" << endl<< endl;

    cout<<"Computing Share Memory + Ballot scan: ";
    t = kcoreSharedMemBallot(data_graph);
    cout<<"Done: "<< t << "ms" << endl<< endl;

    cout<<"Computing Share Memory + Efficient scan: ";
    t = kcoreSharedMemEfficient(data_graph);
    cout<<"Done: "<< t << "ms" << endl<< endl;
    
    return 0;
}
