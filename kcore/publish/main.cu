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
void repSimulation(int (*kern)(T), Graph& g){
    float sum=0;
    int rep = 10;
    for(int i=0;i<rep;i++){
        sum+=(*kern)(g);
    }
    cout<<"EX: "<<sum/rep<<endl;
}

int main(int argc, char *argv[]){
    if (argc < 2) {
        cout<<"Please provide data file"<<endl;
        exit(-1);
    }
    std::string ds = argv[1];

    cout<<"Graph loading Started... "<<endl;    
    Graph g(ds);
    unsigned int t;
    cout<<ds<<endl;
    cout<<"V: "<< g.V<<endl;
    cout<<"E: "<< g.E<<endl;

    cout<<"Computing ours: ";
    repSimulation(kcore, g);
    cout<<"Kmax: "<<g.kmax<<endl;

    cout<<"Computing Ours + Shared Memory: ";
    repSimulation(kcoreSharedMem, g);

    
    cout<<"Computing Ours + Vertex Prefetching: ";
    repSimulation(kcorePrefetch, g);


    cout<<"Computing Efficient scan: ";
    repSimulation(kcoreEfficientScan, g);
    
    cout<<"Computing Ballot scan: " ;
    repSimulation(kcoreBallotScan, g);
    
    cout<<"Computing Efficient scan + Shared Memory + : ";
    repSimulation(kcoreSharedMemEfficient, g);

    cout<<"Computing Ballot scan + Shared Memory: ";
    repSimulation(kcoreSharedMemBallot, g);

    cout<<"Computing Efficient Scan + Vertex Prefetching: ";
    repSimulation(kcoreEfficientScanPrefetch, g);

    cout<<"Computing Ballot scan + Vertex Prefetching: ";
    repSimulation(kcoreBallotScanPrefetch, g);
    
    return 0;
}
