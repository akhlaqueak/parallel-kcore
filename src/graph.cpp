
#include "../inc/graph.h"

Graph::Graph(std::string input_file){
    
    vector< set<unsigned int> > ns;
    V = file_reader(input_file, ns);
    degrees = new unsigned int[V];


    neighbors_offset = new unsigned int[V+1];
    neighbors_offset[0] = 0;
    
    #pragma omp parallel for
    for(int i=0;i<V;++i){
        degrees[i] = ns[i].size();
        neighbors_offset[i+1] = neighbors_offset[i] + ns[i].size();
    }

    E = neighbors_offset[V];
    neighbors = new unsigned int[neighbors_offset[V]];
    // AVG_degrees = E/V + 2;

    unsigned int j = 0;
    for(unsigned int i=0;i<V;++i){
        std::set<unsigned int> s = ns[i];
        for(std::set<unsigned int>::iterator p = s.begin();p!=s.end();p++){
            neighbors[j] = *p;
            j++;
        }
    }
}

Graph::~Graph(){
    // delete [] neighbors;
    // delete [] neighbors_offset;
    // delete [] degrees;
}