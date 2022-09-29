
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

    vector<pair<unsigned int, unsigned int>> edges;
    V = 0;
    while(infile>>s>>t){
        if(s == t) continue; // to remove self loop
        V = max(s,V);
        V = max(t,V);
        edges.push_back({s, t});
    }
    V++; // vertices index start from zero, so number of vertices are 1 greater than largest vertex ID
    degrees = new unsigned int[V];
    unsigned int* tempOffset = new unsigned int[V];

    #pragma omp parallel for
    for(int i=0;i<V;++i){
        degrees[i] = 0;
        tempOffset[i] = 0;
    }

    for(auto &edge : edges){
        degrees[edge.first]++;
        degrees[edge.second]++;
    }

    neighbors_offset = new unsigned int[V+1];

    neighbors_offset[0] = 0;
    partial_sum(degrees, degrees+V, neighbors_offset+1);

    E = neighbors_offset[V];
    neighbors = new unsigned int[E];

    for(int i=0;i<V;i++){
        tempOffset[i] = neighbors_offset[i];
    }

    unsigned int index;
    // #pragma omp parallel for
    for(auto &edge : edges){
        s = edge.first;
        t = edge.second;

        index = tempOffset[s]++;
        neighbors[index] = t;

        index = tempOffset[t]++;
        neighbors[index] = s;
    }

    // for(int i=0;i<V;i++){
    //     sort(neighbors + neighbors_offset[i], neighbors+neighbors_offset[i+1]);
    // }
    delete [] tempOffset;
}

void Graph::writeKCoreToDisk(std::string file){
    // writing kcore in json dictionary format
    std::ofstream out(OUTPUT_LOC + string("pkc-kcore-") + file);

    out<<"{ ";
   
    for(unsigned long long int i=0;i<V;++i)
            // not writing zero degree nodes, because certain nodes in dataset are not present... 
            // our algo treats them isloated nodes, but nxcore doesn't recognize them
        if(degrees[i]!=0)
           out<<'"'<<i<<'"'<<": "<<degrees[i]<<", "<<endl;
    out.seekp(-3, ios_base::end);
    out<<" }";
    out.close();
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