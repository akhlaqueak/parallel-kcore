
#ifndef COMMON_H
#define COMMON_H

#define DS_LOC string("../data_set/data/")
#define OUTPUT_LOC string("../output/")

#include <iostream>
#include <vector>
#include <set>
#include <fstream>
#include <numeric>
#include <vector>
#include <iterator>
#include <functional>

#include "omp.h"
using namespace std;

typedef struct G_pointers {
    unsigned int* neighbors;
    unsigned int* neighbors_offset;
    unsigned int* degrees;
    unsigned int V;
} G_pointers;//graph related


#endif //CUTS_COMMON_H