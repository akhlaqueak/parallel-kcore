
#ifndef CUTS_HOST_FUNCS_H
#define CUTS_HOST_FUNCS_H
#include "./gpu_memory_allocation.h"
#include "./device_funcs.h"
void find_kcore(string data_file,bool write_to_disk);
void find_kcore_CPU(string data_file,bool write_to_disk);

#endif //CUTS_HOST_FUNCS_H
