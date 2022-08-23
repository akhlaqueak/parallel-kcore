

#ifndef CUTS_GRAPH_H
#define CUTS_GRAPH_H
#include "./util.h"
class Graph{
public:
    unsigned int V;
    unsigned int bufTail;
    unsigned int AVG_DEGREE = 0;
    unsigned int * neighbors;
    unsigned int * neighbors_offset;
    unsigned int * degrees;
    Graph(std::string input_file);
    ~Graph();
};
#endif //CUTS_GRAPH_H