
#include "../inc/graph.h"
bool Graph::readSerialized(string input_file){
    ifstream file;
    file.open(string(OUTPUT_LOC) + string("serialized-") + input_file);
    if(file){
        cout<<"Reading serialized file... "<<endl;
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
        file.close();
        return true;
    }else{
        cout<<"readSerialized: File couldn't open"<<endl;
    }

    return false;
}

void Graph::writeSerialized(string input_file){

    ofstream file;
    file.open(string(OUTPUT_LOC) + string("serialized-") + input_file);
    if(file){
        file<<V<<endl;
        file<<E<<endl;
        for(int i=0;i<V;i++)
            file<<degrees[i]<<endl;
        for(int i=0;i<V+1;i++)
            file<<neighbors_offset[i]<<' ';
        for(int i=0;i<E;i++)
            file<<neighbors[i]<<' ';
        file.close();
    }
    else{
        cout<<"writeSerialized: File couldn't open"<<endl;
    }
}

void Graph::readFile(string input_file){
    vector< set<unsigned int> > ns;
    V = file_reader(input_file, ns);
    cout<<"v: "<<V<<endl;
    degrees = new unsigned int[V];


    cout<<"degree allocated: "<<V<<endl;
    
    #pragma omp parallel for
    for(int i=0;i<V;++i){
        degrees[i] = ns[i].size();
    }
    cout<<"degree populated: "<<V<<endl;
    neighbors_offset = new unsigned int[V+1];
    cout<<"neighbors offset allocated: "<<V<<endl;

    neighbors_offset[0] = 0;
    partial_sum(degrees, degrees+V, neighbors_offset+1);
    cout<<"Error in partial sum: "<<V<<endl;

    E = neighbors_offset[V];
    neighbors = new unsigned int[E];
    cout<<"neighbors allocated: "<<E<<endl;

    #pragma omp parallel for
    for(int i=0;i<V;i++){
        auto it = ns[i].begin();
        for(int j=neighbors_offset[i]; j < neighbors_offset[i+1]; j++, it++)
            neighbors[j] = *it;
    }
    cout<<"It's last line: "<<V<<endl;

}

Graph::Graph(std::string input_file){
    if(readSerialized(input_file)) return;
    cout<<"Reading normal file... "<<endl;

    readFile(input_file);
    writeSerialized(input_file);
}

Graph::~Graph(){
    cout<<"Deallocated... "<<endl;
    delete [] neighbors;
    delete [] neighbors_offset;
    delete [] degrees;
}