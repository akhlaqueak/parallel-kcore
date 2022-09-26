#include "../inc/util.h"

unsigned  int file_reader(std::string input_file, vector<set<unsigned int>> &ns){
    double load_start = omp_get_wtime();
    ifstream infile;
    infile.open(DS_LOC + input_file);
    if(!infile){
        cout<<"load graph file failed "<<endl;
        exit(-1);
    }
    unsigned int V, s, t;

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


    ns = vector<set<unsigned int>>(V);

    cout<<"readfile: "<<V<<endl;
    unsigned int v = 0;


    while(infile>>s>>t){
        // if(s == t) continue; // it's a self loop
        // v = max(s, v);
        // v = max(t, v);
        if(s>=V) cout<<s<<endl;
        if(t>=V) cout<<t<<endl;
        
        assert(s<V);
        assert(t<V);

        ns[s].insert(t);
        ns[t].insert(s);
    }

    infile.close();
    double load_end = omp_get_wtime();
    cout<<"Loaded in: "<<load_end-load_start<<", max v: "<<v<<endl;
    return V;
}

void write_kcore_to_disk(unsigned int *degrees, unsigned long long int V, std::string file){
    // writing in json dictionary format
    std::ofstream out(OUTPUT_LOC + string("pkc-kcore-") + file);
    // first entry is read as zero degree node by networkx, 
    // to make it compatible just insert this dummy entry
    out<<"{ ";
    // out<<'"'<<V<<'"'<<": "<<0; 
    
    for(unsigned long long int i=0;i<V;++i)
        if(degrees[i]!=0)
            // not writing zero degree nodes, because certain nodes in dataset are not present... 
            // our algo treats them isloated nodes, but nxcore doesn't recognize them
           out<<'"'<<i<<'"'<<": "<<degrees[i]<<", "<<endl;
    out.seekp(-3, ios_base::end);
    out<<" }";
    out.close();
}
