# K-Core implementation as given in : 
# Parallel k-core Decomposition on Multicore Platforms
# https://ieeexplore.ieee.org/document/7965211

Run the code by the following command, which will pull from github, and runs the code:


` nvcc main.cu src/* -lgomp -o gpu_bk.out 
` ./gpu_bk.out data_set/data/ours_format/Enron.g