import os
import time
import sys
import subprocess as sp
import numpy as np
# This file converts the graphs available at 
# 1. konect.cc
# 2. stanford snap
# 3. https://sparse.tamu.edu/
DSLOC = "data/"
def KonnectDataset(url):
    # URL: http://konect.cc/files/download.tsv.dblp-author.tar.bz2
    # dataset zip file: download.tsv.dblp-author.tar.bz2
    # dataset extracted file: dataset/out.dblp-author
    zipfile = url.split("/")[-1]
    extfile = zipfile.split(".")[2]
    txtfile = DSLOC + extfile + ".txt"
    extfile = extfile + "/out." + extfile
    if not os.path.isfile(zipfile):
        sp.run(["wget", url])
    if not os.path.isfile(extfile):
        sp.run(["tar", "-xvf", zipfile])
    data = np.loadtxt(extfile, comments="%", usecols=(0, 1))
    data = data[data[:, 0]!=data[:,1]] # removing self loops
    V = np.max(data) + 1
    np.savetxt(txtfile, data, fmt='%i', header=str(V))
    print(data.shape)
    
def SnapDataset(url):
    # URL: https://snap.stanford.edu/data/as-skitter.txt.gz
    # dataset zip file: as-skitter.txt.gz
    # extract file: as-skitter.txt
    zipfile = url.split("/")[-1]
    extfile = zipfile.split(".")[0] + ".txt"
    txtfile = DSLOC + extfile
    if not os.path.isfile(zipfile):
        sp.run(["wget", url])    
    if not os.path.isfile(extfile):
        sp.run(["gunzip", zipfile])
    data = np.loadtxt(extfile, comments="#", usecols=(0, 1))
    data = data[data[:, 0]!=data[:,1]] # removing self loops
    V = np.max(data) + 1
    np.savetxt(txtfile, data, fmt='%i', header=str(V))
    print(data.shape)

def HerokuappDataset(url):
    # URL: https://suitessplit-collection-website.herokuapp.com/MM/SNAP/soc-LiveJournal1.tar.gz
    # zipfile: soc-LiveJournal1.tar.gz
    # extract fiel: soc-LiveJournal1/soc-LiveJournal1.mat
    zipfile = url.split("/")[-1]
    extfile = zipfile.split(".")[0]
    txtfile = DSLOC + extfile + ".txt"
    extfile = extfile + "/" + extfile + ".mtx"
    if not os.path.isfile(zipfile):
        sp.run(["wget", url])
    if not os.path.isfile(extfile):
        sp.run(["tar", "-xvf", zipfile])
    data = np.loadtxt(extfile, comments="%", usecols=(0, 1))
    data = data[data[:, 0]!=data[:,1]] # removing self loops
    V = np.max(data) + 1
    np.savetxt(txtfile, data, fmt='%i', header=str(V))
    print(data.shape)

def readFile(file):
    with open(file, "r") as f:
        return f.read().splitlines()

if __name__ == "__main__":
    urls = readFile(sys.argv[1])
    for url in urls:
        if "konect.cc" in url:
            KonnectDataset(url)
        elif "snap.stanford.edu" in url:
            SnapDataset(url)
        elif "herokuapp.com" in url:
            HerokuappDataset(url)
