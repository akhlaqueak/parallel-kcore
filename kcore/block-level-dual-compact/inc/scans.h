#ifndef SCANS_H
#define SCANS_H

#include "../inc/common.h"

__device__ void scanBlockHillis(unsigned int* addresses);

__device__ void scanBlockBelloch(unsigned int* addresses);

__device__ void scanWarpHillis(unsigned int* addresses);

__device__ void scanWarpBelloch(unsigned int* addresses);

#endif //SCANS_H
