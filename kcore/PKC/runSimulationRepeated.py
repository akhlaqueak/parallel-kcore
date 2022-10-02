import os
from re import VERBOSE
import subprocess as sp
import sys
import json
from unittest import result
import networkx as nx
from subprocess import PIPE
import statistics as stat


# datasets = ["Enron.g", "wikipedia-link-de.g", "trackers.g", "soc-Journal.g", \
#     "dblp-author.g", "patentcite.g", "soc-pokec-relationships.g", "wikiTalk.g", "twitter_mpi.g"\
#     ]
# datasets = ['amazon0601.txt',   'cit-Patents.txt',     'in-2004.txt',         'patentcite.txt',        'uk-2002.txt',  'web-BerkStan.txt',  'wiki-Talk.txt', 'arabic-2005.txt',  'dblp-author.txt',     'indochina-2004.txt',  'uk-2005.txt',  'web-Google.txt', 'as-skitter.txt',   'hollywood-2009.txt', 'it-2004.txt',         'soc-LiveJournal1.txt',  'wb-edu.txt',   'webbase-2001.txt']
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
OUTPUT = "../output/"
DATASET = "../data_set/data/"
VERIFY = False
VERBOSE = True
NITERATIONS = 1
for ds in datasets:
    print(ds, ": Started... ", end=" ", flush=True)
    # OMP_NUM_THREADS=32 ./pkc.exe ../data_set/data/in-2004.txt 
    os.environ['OMP_NUM_THREADS'] = "32"
    sp.run([ "./pkc.exe", "../data_set/data/" + ds])




