#include "../inc/util.h"

unsigned  int file_reader(std::string input_file, vector<set<unsigned int>> &ns){
    double load_start = omp_get_wtime();
    ifstream infile;
    infile.open("../data_set/data/ours_format/" + input_file);
    if(!infile){
        cout<<"load graph file failed "<<endl;
        exit(-1);
    }
    unsigned int V, s, t;

    infile>>V;

    ns = vector<set<unsigned int>>(V);

    while(infile>>s>>t){
        ns[s].insert(t);
        ns[t].insert(s);
    }
    infile.close();
    double load_end = omp_get_wtime();
    return V;
}

void write_kcore_to_disk(unsigned int *degrees, unsigned long long int V, std::string file){
    // writing in json dictionary format
    std::ofstream out("../output/" + file + "-pkc-kcore");
    out<<"{ ";
    out<<'"'<<V<<'"'<<": "<<0; // first entry is read as zero degree node by networkx, to make it compatible just insert this dummy entry
    for(unsigned long long int i=0;i<V;++i){
        if(degrees[i]!=0){
            
            // not writing zero degree nodes, because certain nodes in dataset are not present... 
           out<<', "'<<i<<'"'<<": "<<degrees[i];
        }
    }

    out<<" }";
    out.close();
}
