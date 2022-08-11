#include "../inc/util.h"

unsigned  int file_reader(std::string input_file, vector<set<unsigned int>> &ns){
    double load_start = omp_get_wtime();
    ifstream infile;
    infile.open("../data_set/data/ours_format/" + input_file);
    if(!infile){
        cout<<"load graph file failed "<<endl;
        exit(-1);
    }
    unsigned int V  = 0;
    string line;
    const std::string delimter = "\t";
    unsigned int line_index = 0;
    getline(infile,line);
    V = stoi(line);
    for(unsigned int i=0;i<V;++i){
        set<unsigned int> temp_set;
        ns.push_back(temp_set);
    }

    while(getline(infile,line)){
        auto pos = line.find(delimter);
        if(pos == std::string::npos){
            continue;
        }
        int s = stoi(line.substr(0, pos));
        int t = stoi(line.substr(pos + 1, line.size() - pos - 1));
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

    for(unsigned long long int i=0;i<V;++i){
        if(degrees[i]!=0)
           out<<string(i)<<": "<<degrees[i]","<<endl;
    }

    out<<" }";
    out.close();
}
