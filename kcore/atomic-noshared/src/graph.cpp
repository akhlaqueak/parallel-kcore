
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

    double load_start = omp_get_wtime();
    ifstream infile;
    infile.open(DS_LOC + input_file);
    if(!infile){
        cout<<"load graph file failed "<<endl;
        exit(-1);
    }
    unsigned int s, t;

/**
 * @brief Dataset format:
 * # Number of nodes
 * source destination
 * source destination
 * source destination
 * source destination
 * 
 */
    char dumy;
    infile>>dumy; // to read # in the first line... 
    infile>>V;
    V++;

    vector<pair<unsigned int, unsigned int>> edges;

    while(infile>>s>>t){
        assert(s<V);
        assert(t<V);
        if(s == t) continue; // to remove self loop
        edges.push_back({s, t});
    }
    degrees = new unsigned int[V];
    unsigned int* tempOffset = new unsigned int[V];


    cout<<"degree allocated: "<<V<<endl;
    
    #pragma omp parallel for
    for(int i=0;i<V;++i){
        degrees[i] = 0;
        tempOffset[i] = 0;
    }

    cout<<"degrees initialized: "<<V<<endl;
    for(auto &edge : edges){
        degrees[edge.first]++;
        degrees[edge.second]++;
    }

    neighbors_offset = new unsigned int[V+1];
    cout<<"neighbors offset allocated: "<<V<<endl;

    neighbors_offset[0] = 0;
    partial_sum(degrees, degrees+V, neighbors_offset+1);
    cout<<"in partial sum: "<<V<<endl;

    E = neighbors_offset[V];
    neighbors = new unsigned int[E];
    cout<<"neighbors allocated: "<<E<<endl;

    // #pragma omp parallel for
    for(auto &edge : edges){
        cout<<s<<","<<t<<":";
        s = edge.first;
        t = edge.second;
        assert(s<V);
        assert(t<V);
        index = neighbors_offset[s] + tempOffset[s];
        assert(index<E);
        neighbors[index] = t;
        tempOffset[s]++;

        index = neighbors_offset[t] + tempOffset[t];
        assert(index<E);
        neighbors[index] = s;
        tempOffset[t]++;
    }
    cout<<"It's last line: "<<V<<endl;

    // for(int i=0;i<V;i++){
    //     sort(neighbors + neighbors_offset[i], neighbors+neighbors_offset[i+1]);
    // }
    cout<<"************"<<endl;


    delete [] tempOffset;
}

Graph::Graph(std::string input_file){
    // if(readSerialized(input_file)) return;
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