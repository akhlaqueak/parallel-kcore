import os
import time
import sys
import subprocess as sp
import numpy as np
# This file converts the graphs available at 
# 1. konect.cc
# 2. stanford snap
# 3. https://sparse.tamu.edu/

# it removes cycles in the graph, and create the graph in 
# (first line is number of vertices, followed by edge at every line)

# V
# source destination
# source destination
# source destination
# source destination
# source destination
DOWNLOAD = True
def convert_format(arg):
    in_file = arg[1]
    graph = in_file.split('.')[0] + ".g"

    comment_ch = '#@%!'
    st = time.time()
    lines = []
    out_lines = []
    max_node = 0

    with open(in_file,'r') as reader:
        lines = reader.readlines()

    for line in lines:
        if line[0] in comment_ch:
            continue
        s, e = line.split()
        if s == e: # it's a self-loop.
            continue
        out_lines.append(line)
        max_node = max(max_node, int(s), int(e))

    with open(graph, 'w') as writer:
        writer.write(str(max_node+1)+'\n')
        writer.writelines(out_lines)

    print("Elapsed: ", time.time()-st)
def KonnectDataset(url):
    # URL: http://konect.cc/files/download.tsv.dblp-author.tar.bz2
    # dataset zip file: download.tsv.dblp-author.tar.bz2
    # dataset extracted file: dataset/out.dblp-author
    zipfile = url.split("/")[-1]
    sp.run(["tar", "-xvf", zipfile])
    extfile = zipfile.split(".")[2]
    extfile = extfile + "/out." + extfile
    data = np.loadtxt(extfile, comments="%")
    
def SnapDataset(url):
    # URL: https://snap.stanford.edu/data/as-skitter.txt.gz
    # dataset zip file: as-skitter.txt.gz
    # extract file: as-skitter.txt
    zipfile = url.split("/")[-1]
    sp.run(["gunzip", zipfile])
    extfile = zipfile.split(".")[0] + ".txt"   
    data = np.loadtxt(extfile, comments="#")

def HerokuappDataset(url):
    # URL: https://suitessplit-collection-website.herokuapp.com/MM/SNAP/soc-LiveJournal1.tar.gz
    # zipfile: soc-LiveJournal1.tar.gz
    # extract fiel: soc-LiveJournal1/soc-LiveJournal1.mat
    zipfile = url.split("/")[-1]
    sp.run(["tar", "-xvf", zipfile])
    extfile = zipfile.split(".")[0]
    extfile = extfile + "/" + extfile + ".mat"
    data = np.loadtxt(extfile, comments="%")
    np.delete(data, 1)
    print(data.shape)

def readFile(file):
    with open(file, "r") as f:
        return f.readlines()

if __name__ == "__main__":
    urls = readFile(sys.argv[1])
    for url in urls:
        if DOWNLOAD:
            sp.run(["wget", url]) 
        if "konect.cc" in url:
            KonnectDataset(url)
        elif "snap.stanford.edu" in url:
            SnapDataset(url)
        elif "herokuapp.com" in url:
            HerokuappDataset(url)



    # convert_format(sys.argv)