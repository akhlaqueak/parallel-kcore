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