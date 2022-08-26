import os
import subprocess as sp
import sys
import json
import networkx as nx
from subprocess import PIPE

FOLDERS = ["ballotcompact-linkedlist"]
OUTPUT = "../output/"
DATASET = "../data_set/data/ours_format/"
VERIFY = False

def verify(dataset):
    nx_kcore = {}
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
        print(dataset, "Verification Test Passed!")
    else:
        print(dataset, "Verification Test Failed!")
        print("The difference is: ")
        diff = set(nx_kcore.items()) ^ set(pkc_kcore.items())
        print (diff)

def parseResult(output):
    # One of the line in output has this format
    # Elapsed Time: 69.8055
    line = [x for x in output.split("\n") if "Elapsed Time" in x]
    return line[0].split(":")[1]

def parseDataSet(args):
    datasets = ("Enron.g", "wikipedia-link-de.g", "trackers.g", "soc-Journal.g", "dblp-author.g", "patentcite.g", "soc-pokec-relationships.g", "wikiTalk.g")
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
    elif arg in datasets:
        ds.append(arg) 
    else:
        print("Please provide valid dataset: ", [(x, y) for x, y in enumerate(datasets)])
        exit(0)
    return ds

def runSim(datasets):
    results = []
    for ds in datasets:
        print(ds, ": Started... ", end=" ", flush=True)
        output = sp.run(["./kcore", ds], stdout=PIPE, stderr=PIPE)
        time = parseResult(output.stdout.decode()) # decode is converting byte string to regular
        results.append((ds, time),)
        print("Completed")
    return results

def parseFolder(args):
    folders = ["atomic", "fastcompact", "fastcompact-warponly", "ballotcompact", "ballotcompact-warponly", "ballotcompact-linkedlist"]
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
    return folder
if __name__ == "__main__":
    datasets = parseDataSet(sys.argv)
    folders = parseFolder(sys.argv)
    for folder in folders:
        os.chdir(folder)
        sp.run(["make"])
        print("Executing in ", folder)
        results = runSim(datasets)
        if VERIFY:
            for ds in datasets:
                verify(ds)
        print("### Results for ", folder, " ###")
        for ds, time in results:
            print(ds, time)
        print("### --------------- ###")
        os.chdir("../")