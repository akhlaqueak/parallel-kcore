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
template<class T>
void invoke(Graph& g, int (*kern)(T)){
    float sum=0;
    for(int i=0;i<5;i++){
        sum+=(*kern)(g);
    }
    cout<<"EX: "<<sum/5.0<<endl;
}

int main(int argc, char *argv[]){
    if (argc < 2) {
        cout<<"Please provide data file"<<endl;
        exit(-1);
    }
    std::string data_file = argv[1];

    cout<<"Graph loading Started... "<<endl;    
    Graph g(data_file);
    unsigned int t;

    cout<<"V: "<< g.V<<endl;
    cout<<"E: "<< g.E<<endl;

    // cout<<"Computing ours... ";
    t = kcore(g);
    cout<<"Kmax: "<<g.kmax<<endl;
    cout<<"Our algo Done: "<< t  << endl<< endl;

    cout<<"Computing Shared Memory + Ours... ";
    t = kcoreSharedMem(g);
    cout<<"Done: "<< t  << endl<< endl;
    
    cout<<"Computing Vertex Prefetching + Ours ... ";
    t = kcorePrefetch(g);
    cout<<"Done: "<< t  << endl<< endl;

    cout<<"Computing Efficient scan: ";
    t = kcoreEfficientScan(g);
    cout<<"Done: "<< t  << endl<< endl;
    
    cout<<"Computing Ballot scan: " ;
    t = kcoreBallotScan(g);
    cout<<"Done: "<< t  << endl<< endl;
    
    cout<<"Computing Ballot scan + Vertex Prefetching: ";
    t = kcoreBallotScanPrefetch(g);
    cout<<"Done: "<< t  << endl<< endl;

    cout<<"Computing Efficient Scan, Vertex Prefetching: ";
    t = kcoreEfficientScanPrefetch(g);
    cout<<"Done: "<< t  << endl<< endl;

    cout<<"Computing Share Memory + Ballot scan: ";
    t = kcoreSharedMemBallot(g);
    cout<<"Done: "<< t  << endl<< endl;

    cout<<"Computing Share Memory + Efficient scan: ";
    t = kcoreSharedMemEfficient(g);
    cout<<"Done: "<< t  << endl<< endl;
    invoke(g, kcoreSharedMemEfficient);
    return 0;
}
