import os
from re import VERBOSE
import subprocess as sp
import sys
import json
from unittest import result
import networkx as nx
from subprocess import PIPE
import statistics as stat

folders = ["atomic", "fastcompact", "fastcompact-warponly", "ballotcompact", \
        "ballotcompact-warponly", "ballotcompact-linkedlist", "ballotcompact-warponly-sep-kernels", "cpu", 
        "atomic-linked-list-separate-kernels", "compact-linked-list-separate-kernels"]

datasets = ["Enron.g", "wikipedia-link-de.g", "trackers.g", "soc-Journal.g", \
    "dblp-author.g", "patentcite.g", "soc-pokec-relationships.g", "wikiTalk.g", "twitter_mpi.g"\
    ]

OUTPUT = "../output/"
DATASET = "../data_set/data/ours_format/"
VERIFY = True
VERBOSE = True
NITERATIONS = 10

def verify(datasets):
    difference = {}
    for dataset in datasets: 
        nx_kcore = {}
        print("Verifying ", dataset, "... ", flush=True, end=" ")
        try:
            file = open(OUTPUT + "nx-kcore-" + dataset, 'r')
            nx_kcore = json.load(file)
            file.close()
        except IOError:
            G = nx.read_adjlist(DATASET + dataset)
            nx_kcore = nx.core_number(G)
            # save the file for future use... 
            json.dump(nx_kcore, open(OUTPUT + "nx-kcore-" + dataset, 'w'))

        pkc_kcore = json.load(open(OUTPUT + "pkc-kcore-" + dataset, 'r'))

        if nx_kcore == pkc_kcore:
            difference[dataset] = 0
            print(" Passed!")
        else:
            print(" Failed!", end=" ")
            print("The difference is: ", flush=True, end=" ")
            diff = set(nx_kcore.items()) ^ set(pkc_kcore.items())
            print (len(diff)/2, " items")
            difference[dataset] = len(diff)/2
    return difference

def parseResult(output):
    # One of the line in output has this format
    # Elapsed Time: 69.8055
    line = [x for x in output.split("\n") if "Elapsed Time" in x]
    return line[0].split(":")[1]

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

def runFolder(datasets):

    results = {}
    for ds in datasets:
        print(ds, ": Started... ", end=" ", flush=True)
        output = sp.run(["./kcore", ds], stdout=PIPE, stderr=PIPE)
        text = output.stdout.decode()
        if(VERBOSE): 
            print(text)
        time = parseResult(text) # decode is converting byte string to regular
        results[ds] = time
        print("Completed")
    return results

def parseFolder(args):
    folder = []
    arg = "" if len(args)<3 else args[2]

    if arg == "":
        folder.append(folders[0])
    elif arg.isnumeric():
        ind = int(arg)
        if(ind<len(folders)):
            folder.append(folders[int(arg)])
        else:
            print("Please provide valid folder: ", [(x, y) for x, y in enumerate(folders)])
    elif arg == "all":
        folder = folders
    elif arg in folders:
        folder.append(arg) 
    else:
        print("Please provide valid folder: ", [(x, y) for x, y in enumerate(folders)])
        exit(0)
    return folder[0]

def runSimulation(datasets, folder):
    verResult = {}
    execTime = {}
    os.chdir(folder)
    sp.run(["make"])
    print("Executing in ", folder)
    execTime = runFolder(datasets)
    if VERIFY:
        verResult = verify(datasets)
    os.chdir("../")

    return execTime, verResult

def repeatedSimulations(datasets, folder):
    execTime = {}
    verResult = {}
    for ds in datasets:
        execTime[ds] = []
        verResult[ds] = []
    
    for i in range(NITERATIONS):
        print("Running Simulation No. ", i+1)
        exec, ver = runSimulation(datasets, folder)
        for ds in datasets:
            execTime[ds].append(int(exec[ds]))
            verResult[ds].append(ver[ds])
    for ds in datasets:
        print(ds, execTime[ds])
        print(ds, verResult[ds])
        print(ds, "Execution Time Average : ", stat.mean(execTime[ds]))
        print(ds, "Difference Average : ", stat.mean(verResult[ds]))
        

if __name__ == "__main__":
    sp.run(["git", "pull"])
    if len(sys.argv) == 1:
        print("usage: ")
        print("python runSimulation.py datasetIndex|all folderIndex \n")
        print("datasetIndex: ", [str(x)+ ": " + y for x, y in enumerate(datasets)])
        print("folderIndex: ", [str(x)+ ": " + y  for x, y in enumerate(folders)])
        exit(0)
    datasets = parseDataSet(sys.argv)
    folder = parseFolder(sys.argv)

    repeatedSimulations(datasets, folder)



    
# max=10
# for i in `seq 1 $max`
# do
#     python runSimulation.py all 6
# done