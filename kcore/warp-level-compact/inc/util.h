

#ifndef CUTS_UTIL_H
#define CUTS_UTIL_H
#include "./common.h"
unsigned int file_reader(std::string input_file, vector<set<unsigned int>> &ns);

void write_to_disk(unsigned int*, unsigned long long int, std::string file);
#endif //CUTS_UTIL_H
