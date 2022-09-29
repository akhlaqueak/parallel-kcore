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
# datasets = ['amazon0601.txt',   'cit-Patents.txt',     'in-2004.txt',         'patentcite.txt']

folders = ['preallocated-final']

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
OUTPUT = "../output/"
DATASET = "../data_set/data/"
VERIFY = False
VERBOSE = True
NITERATIONS = 3

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
    mem = {}
    for ds in datasets:
        print(ds, ": Started... ", end=" ", flush=True)
        memp = sp.Popen("../mem/mem.sh", stdout=PIPE, stderr=PIPE)
        output = sp.run(["./kcore", ds], stdout=PIPE, stderr=PIPE)
        memp.kill()
        memtrace, errtrace = memp.communicate()
        memtrace = list(map(int, memtrace.split())) # split and convert to integers

        mem[ds] = max(memtrace)
        text = output.stdout.decode()
        if(VERBOSE): 
            print(text)
        time = parseResult(text) # decode is converting byte string to regular
        results[ds] = time
        print("Completed")
    return results, mem

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
    else:
        folder.append(arg) 
    return folder[0]

def runSimulation(datasets, folder):
    verResult = {}
    execTime = {}
    os.chdir(folder)
    sp.run(["make"])
    print("Executing in ", folder)
    execTime, mem = runFolder(datasets)
    if VERIFY:
        verResult = verify(datasets)
    os.chdir("../")

    return execTime, mem, verResult

def repeatedSimulations(datasets, folder):
    execTime = {}
    verResult = {}
    memtraces = {}
    for ds in datasets:
        execTime[ds] = []
        verResult[ds] = []
        memtraces[ds] = []
    
    for i in range(NITERATIONS):
        print("Running Simulation No. ", i+1)
        exec, mem, ver = runSimulation(datasets, folder)
        for ds in datasets:
            execTime[ds].append(float(exec[ds]))
            memtraces[ds].append(mem[ds])
            if VERIFY:
                verResult[ds].append(ver[ds])
    print("###---------------###")
    print("###- Execution Time -###")
    
    for ds in datasets:
        print(ds, end=" ")
        for t in execTime[ds]:
            print(t, end=" ")
        print(round(stat.mean(execTime[ds])), end=" ")
        print(max(memtraces[ds]))
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
    folder = parseFolder(sys.argv)

    repeatedSimulations(datasets, folder)

