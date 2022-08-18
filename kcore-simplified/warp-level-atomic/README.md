# K-Core implementation

GPU implementation of: 

H. Kabir and K. Madduri, "Parallel k-Core Decomposition on Multicore Platforms," 2017 IEEE International Parallel and Distributed Processing Symposium Workshops (IPDPSW), 2017, pp. 1482-1491, doi: 10.1109/IPDPSW.2017.151.

Run the code by the following command, which will pull from github, and runs the code:


` nvcc main.cu src/* -lgomp -o gpu_bk.out `

`./gpu_bk.out data_set/data/ours_format/Enron.g`