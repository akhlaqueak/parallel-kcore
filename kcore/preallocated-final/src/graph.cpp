
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
 */// read number of nodes... 
    string line;
    vector<pair<unsigned int, unsigned int>> lines;
    streampos oldpos;
    while(true){
        oldpos = infile.tellg();
        // ignore all comments lines assumint they contain # or %
        getline(infile, line);
        if(line.find('#') == string::npos && line.find('%') == string::npos) break;
    }

    if(input_file.find(".mtx")!=string::npos){
        // first data line of mtx file is also ignored... 
        infile.seekg(oldpos);
    }



    V = 0;
    while(infile>>s>>t){
        if(s==t) continue; // remove self loops
        V = max(s, V);
        V = max(t, V);
        lines.push_back({s,t});
    }
    infile.close();

    V++; // vertices index starts from 0, so add 1 to number of vertices.

    vector<set<unsigned int>> ns(V);
    
    for(auto &p : lines){
        ns[p.first].insert(p.second);
        ns[p.second].insert(p.first);
    }
    
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

    auto start = chrono::steady_clock::now();
    readFile(input_file);
    
    cout<<"File Loaded in: " <<
    chrono::duration_cast<chrono::milliseconds>(end - start).count()
    <<endl;

    writeSerialized(input_file);
}

Graph::~Graph(){
    cout<<"Deallocated... "<<endl;
    delete [] neighbors;
    delete [] neighbors_offset;
    delete [] degrees;
}