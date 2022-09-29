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
'arabic-2005.txt', \
'as-skitter.txt', \
'dblp-author.txt', \
'hollywood-2009.txt', \
'in-2004.txt', \
'indochina-2004.txt', \
'it-2004.txt', \
'patentcite.txt', \
# 'sk-2005.txt', \
'soc-LiveJournal1.txt', \
'trackers.txt', \
'uk-2002.txt', \
'uk-2005.txt', \
'wb-edu.txt', \
'webbase-2001.txt', \
'web-BerkStan.txt', \
'web-Google.txt', \
'wikipedia-link-de.txt', \
'wiki-Talk.txt', \
]
# datasets = ['amazon0601.txt',   'cit-Patents.txt',     'in-2004.txt',         'patentcite.txt']

OUTPUT = "../output/"
DATASET = "../data_set/data/"
VERIFY = False
VERBOSE = True

os.chdir(sys.argv[1])

for ds in datasets:

    print(ds, ": Started... ", end=" ", flush=True)
    memp = sp.Popen("../mem/mem.sh", stdout=PIPE, stderr=PIPE)
    output = sp.run(["./kcore", ds], stdout=PIPE, stderr=PIPE)
    memp.kill()
    memtrace, errtrace = memp.communicate()
    memtrace = list(map(int, memtrace.split())) # split and convert to integers

    mem = max(memtrace)
    text = output.stdout.decode()
    if(VERBOSE): 
        print(text)
    print(ds, " MEM: ", mem)
