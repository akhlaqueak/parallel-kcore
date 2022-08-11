
#ifndef CUTS_COMMON_H
#define CUTS_COMMON_H
#define BLK_NUMS 10
#define BLK_DIM 1024
#define WARPS_EACH_BLK (BLK_DIM/32)
#define WORK_UNITS (BLK_NUMS*WARPS_EACH_BLK)
#define MAX_NV 6000
#define N_THREADS (BLK_DIM*BLK_NUMS)
#define GLBUFFER_SIZE 10000000
#define THID threadIdx.x
#define WARP_SIZE 32
#define UINT unsigned int

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
    unsigned int V;
} G_pointers;//graph related

#endif //CUTS_COMMON_H
