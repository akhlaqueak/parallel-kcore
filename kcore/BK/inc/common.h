
#ifndef CUTS_COMMON_H
#define CUTS_COMMON_H
#define BLK_NUMS 56
#define BLK_DIM 1024
#define WARPS_EACH_BLK (BLK_DIM/32)
#define WORK_UNITS (BLK_NUMS*WARPS_EACH_BLK)
#define MAX_NV 10000
#define BUFF_SIZE 20000
#define N_THREADS (BLK_DIM*BLK_NUMS)
#define GLBUFFER_SIZE 1000000
#define THID threadIdx.x
#define WARP_SIZE 32
#define UINT unsigned int
#define DS_LOC string("../data_set/data/")
#define OUTPUT_LOC string("../output/")
#define REP 1
#define NSUBS 1000
#define SUBG 32
#define WARPID THID>>5
#define LANEID (THID&31)
#define BLKID blockIdx.x
#define FULL 0xFFFFFFFF
// tempsize is max size of a subgraph stored in temp area, in general it should be the size of max degree of supported graph
#define TEMPSIZE 100000
#define R 'r'
#define P 'p'
#define X 'x'
#define Q 'q'

#include <iostream>
#include <vector>
#include <set>
#include <string>
#include <fstream>
#include <map>
#include <utility>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <unordered_map>
#include <stack>
#include <deque>
#include <random>
#include <cuda.h>
#include <bits/stdc++.h>
#include "omp.h"

using namespace std;

typedef struct G_pointers {
    unsigned int* neighbors;
    unsigned int* neighbors_offset;
    unsigned int* degrees;
    unsigned int* degOrder;
    unsigned int V;
    unsigned int* total;
} G_pointers;//graph related

typedef struct Subgraphs{
    unsigned int* offsets;
    unsigned int* vertices;
    char* labels;

    unsigned int* otail;
    unsigned int* ohead;
    unsigned int* vtail;

    unsigned int* tempv;
    unsigned int* templ;
}Subgraphs;


#endif //CUTS_COMMON_H
