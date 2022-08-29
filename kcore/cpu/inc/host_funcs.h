
#ifndef CUTS_HOST_FUNCS_H
#define CUTS_HOST_FUNCS_H
#include "./common.h"
#include "./graph.h"

void processNode(unsigned int v, Graph &g, unsigned int* buffer, unsigned int &tail);

void find_kcore(string data_file,bool write_to_disk);

#endif //CUTS_HOST_FUNCS_H
