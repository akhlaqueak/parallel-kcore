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
datasets = ['amazon0601.txt',   'cit-Patents.txt',     'in-2004.txt',         'patentcite.txt',        'uk-2002.txt',  'web-BerkStan.txt',  'wiki-Talk.txt', 'arabic-2005.txt',  'dblp-author.txt',     'indochina-2004.txt',  'uk-2005.txt',  'web-Google.txt', 'as-skitter.txt',   'hollywood-2009.txt', 'it-2004.txt',         'soc-LiveJournal1.txt',  'wb-edu.txt',   'webbase-2001.txt']

OUTPUT = "../output/"
DATASET = "../data_set/data/"
VERIFY = False
VERBOSE = True
NITERATIONS = 3

def parseResult(output):
    # One of the line in output has this format
    # Elapsed Time: 69.8055
    line = [x for x in output.split("\n") if "PKC time" in x]
    print (output)
    return line[0].split(" ")[2]

def parseDataSet(args):
        #  "twitter.g", "livejournal-groupmemberships.g", "yahoo-song.g", "bag-pubmed.g", \
     	# "dbpedia-link.g", "wikipedia_link_ms.g", "dimacs10-uk-2002.g")
    ds = []
    arg = "" if len(args)<2 else args[1]

    if arg == "":
        ds.append("Enron.g")
    elif arg.isnumeric():
        ind = int(arg)
        if(ind<len(datasets)):
            ds.append(datasets[int(arg)])
        else:
            print("Please provide valid dataset: ", [(x, y) for x, y in enumerate(datasets)])
    elif arg == "all":
        ds = datasets
    else:
        ds.append(arg) 
    return ds

def runSimulation(datasets):

    results = {}
    for ds in datasets:
        print(ds, ": Started... ", end=" ", flush=True)
        # OMP_NUM_THREADS=32 ./pkc.exe ../data_set/data/in-2004.txt 
        output = sp.run([ "./pkc.exe", "../data_set/data/" + ds], stdout=PIPE, stderr=PIPE)
        text = output.stdout.decode()

        if(VERBOSE): 
            print("output: ", text)
        time = parseResult(text) # decode is converting byte string to regular
        results[ds] = time
        print("Completed")
    return results





def repeatedSimulations(datasets):
    execTime = {}
    verResult = {}
    for ds in datasets:
        execTime[ds] = []
        verResult[ds] = []
    
    for i in range(NITERATIONS):
        print("Running Simulation No. ", i+1)
        exec, ver = runSimulation(datasets)
        for ds in datasets:
            execTime[ds].append(float(exec[ds]))
            if VERIFY:
                verResult[ds].append(ver[ds])
    print("###---------------###")
    print("###- Execution Time -###")
    
    for ds in datasets:
        print(ds, end=" ")
        for t in execTime[ds]:
            print(t, end=" ")
        print(stat.mean(execTime[ds]))
    print("###---------------###")
    if VERIFY: 
        print("###- Verification Difference -###")
        for ds in datasets:
            print(ds, end=" ")
            for t in verResult[ds]:
                print(t, end=" ")
            print(stat.mean(verResult[ds]))
        print("###---------------###")
        

if __name__ == "__main__":
    sp.run(["git", "pull"])
    if len(sys.argv) == 1:
        print("usage: ")
        print("python runSimulation.py datasetIndex|all folderIndex \n")
        print("datasetIndex: ", [str(x)+ ": " + y for x, y in enumerate(datasets)])
        print("folderIndex: ", [str(x)+ ": " + y  for x, y in enumerate(folders)])
        exit(0)
    datasets = parseDataSet(sys.argv)

    repeatedSimulations(datasets)

