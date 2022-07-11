/*
 * cuTS:  Scaling Subgraph Isomorphism on Distributed Multi-GPU Systems Using
 *        Trie Based Data Structure
 *
 * Copyright (C) 2021 APPL Laboratories (aravind_sr@outlook.com)
 *
 * This software is available under the MIT license, a copy of which can be
 * found in the file 'LICENSE' in the top-level directory.
 *
 * For further information contact:
 *   (1) Lizhi Xiang (lizhi.xiang@wsu.edu)
 *   (2) Aravind Sukumaran-Rajam (aravind_sr@outlook.com)
 *
 * The citation information is provided in the 'README' in the top-level
 * directory.
 */

#include "../inc/util.h"
unsigned long long int file_reader(std::string input_file, vector<set<unsigned int>> &ns){
    double load_start = omp_get_wtime();
    ifstream infile;
    infile.open(input_file);
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
    std::ofstream out("file");
    out<<V<<endl;

    for(unsigned long long int i=0;i<V;++i){
        out<<i<<" "<<degrees[i]<<endl;
    }

    out.close();
}
