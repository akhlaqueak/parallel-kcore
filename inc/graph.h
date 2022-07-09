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

#ifndef CUTS_GRAPH_H
#define CUTS_GRAPH_H
#include "./score.h"
#include "./util.h"
class Graph{
public:
    unsigned int V;
    unsigned int E;
    unsigned int AVG_DEGREE = 0;
    unsigned int * neighbors;
    unsigned int * neighbors_offset;
    unsigned int * degrees;
    Graph(std::string input_file);
    unsigned long long int file_reader(std::string input_file, vector<set<unsigned int>> &ns);
};
#endif //CUTS_GRAPH_H
