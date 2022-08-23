
#include "../inc/graph.h"
bool Graph::readSerialized(string input_file){
    ifstream file;
    if(file.open(string(DS_LOC) + string("serialized-") + input_file, 'r')){
        file>>V;
        file>>E;
        degrees = new unsigned int[V];
        neighbors_offset = new unsigned int[V+1];
        neighbors = new unsigned int[E];
        for(int i=0;i<V;i++)
            file>>degrees[i];
        for(int i=0;i<V+1;i++)
            file>>neighbors_offset[i];
        for(int i=0;i<E;i++)
            file>>neighbors[i];
        return true;
    }
    file.close();

    return false;
}

void Graph::writeSerialized(string input_file){
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
    ofstream file;
    if(file.open(string(DS_LOC) + string("serialized-") + input_file, 'w')){
        file<<V;
        file<<E;
        for(int i=0;i<V;i++)
            file<<degrees[i]<<' ';
        for(int i=0;i<V+1;i++)
            file<<neighbors_offset[i]<<' ';
        for(int i=0;i<E;i++)
            file<<neighbors[i]<<' ';
    }
    file.close();
}

Graph::Graph(std::string input_file){
    if(readSerialized(input_file)) return;
    
    writeSerialized(input_file);
}

Graph::~Graph(){
    // delete [] neighbors;
    // delete [] neighbors_offset;
    // delete [] degrees;
}