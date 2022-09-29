

#ifndef CUTS_GRAPH_H
#define CUTS_GRAPH_H
#include "./common.h"

class Graph{
public:
    unsigned int V;
    unsigned int E;
    unsigned int AVG_DEGREE = 0;
    unsigned int * neighbors;
    unsigned int * neighbors_offset;
    unsigned int * degrees;
    Graph(std::string input_file);
    bool readSerialized(string input_file);
    void writeSerialized(string input_file);
    void readFile(string input_file);
    ~Graph();
};
#endif //CUTS_GRAPH_H
