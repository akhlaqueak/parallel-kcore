
#include "../inc/graph.h"


Graph::Graph(std::string input_file){
    
    vector< set<unsigned int> > ns;
    V = file_reader(input_file, ns);
    degrees = new unsigned int[V];


    
    #pragma omp parallel for
    for(int i=0;i<V;++i){
        degrees[i] = ns[i].size();
    }

    neighbors_offset = new unsigned int[V+1];
    neighbors_offset[0] = 0;
    partial_sum(degrees, degrees+V, neighbors_offset+1);

    E = neighbors_offset[V];
    neighbors = new unsigned int[E];

    #pragma omp parallel for
    for(int i=0;i<V;i++){
        auto it = ns[i].begin();
        for(int j=neighbors_offset[i]; j < neighbors_offset[i+1]; j++, it++)
            neighbors[j] = *it;
    }

}

Graph::~Graph(){
    // delete [] neighbors;
    // delete [] neighbors_offset;
    // delete [] degrees;
}