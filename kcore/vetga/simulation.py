import os
from re import VERBOSE
import subprocess as sp
import sys
import json
from unittest import result
import networkx as nx
from subprocess import PIPE
import statistics as stat

# folders = ["atomic", "atomic-sepkernel", "atomic-sepkernel-prefetched", "atomic-noshared", "ballotcompact-sepkernel",       "fastcompact", "fastcompact-warponly", "ballotcompact", "ballotcompact-warponly", "ballotcompact-linkedlist", "ballotcompact-warponly-sep-kernels", "cpu",        "atomic-linked-list-separate-kernels", "compact-linked-list-separate-kernels", 
        # "ballotcompact-warponly-sep-kernels-prefetched"]

# datasets = ["Enron.g", "wikipedia-link-de.g", "trackers.g", "soc-Journal.g", \
#     "dblp-author.g", "patentcite.g", "soc-pokec-relationships.g", "wikiTalk.g", "twitter_mpi.g"\
#     ]
# datasets = ['amazon0601.txt',   'cit-Patents.txt',     'in-2004.txt',         'patentcite.txt',        'uk-2002.txt',  'web-BerkStan.txt',  'wiki-Talk.txt', 'arabic-2005.txt',  'dblp-author.txt',     'indochina-2004.txt',  'uk-2005.txt',  'web-Google.txt', 'as-skitter.txt',   'hollywood-2009.txt', 'it-2004.txt',         'soc-LiveJournal1.txt',  'wb-edu.txt',   'webbase-2001.txt']

# folders = ['preallocated-final']

datasets = [
'amazon0601.txt', \
'wiki-Talk.txt', \
'web-Google.txt', \
'web-BerkStan.txt', \
'as-skitter.txt', \
'patentcite.txt', \
'in-2004.txt', \
'dblp-author.txt', \
'wb-edu.txt', \
'soc-LiveJournal1.txt', \
'wikipedia-link-de.txt', \
'hollywood-2009.txt', \
'com-orkut.txt', \
'trackers.txt', \
'indochina-2004.txt', \
'uk-2002.txt', \
'arabic-2005.txt', \
'uk-2005.txt', \
'webbase-2001.txt', \
'it-2004.txt', \
]

datasets = ['amazon0601.txt']
OUTPUT = "../output/"
DATASET = "../data_set/data/"
VERIFY = False
VERBOSE = True

for ds in datasets:

    print(ds, ": Started... ", end=" ", flush=True)
    memp = sp.Popen("./mem.sh", stdout=PIPE, stderr=PIPE)
    output = sp.run(["python3","kcdapp.py", "-m", "torch-gpu", DATASET + ds], stdout=PIPE, stderr=PIPE)
    memp.kill()
    memtrace, errtrace = memp.communicate()
    memtrace = list(map(int, memtrace.split())) # split and convert to integers
    # mem = max(memtrace)
    print(ds, " MEM: ", memtrace)
    text = output.stdout.decode()
    err = output.stderr.decode()
    if(VERBOSE): 
        print(text)
