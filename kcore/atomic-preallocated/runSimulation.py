import os
import subprocess as sp
import sys
from subprocess import PIPE

def parseResult(output):
    # One of the line in output has this format
    # Elapsed Time: 69.8055
    line = [x for x in output.split("\n") if "Elapsed Time" in x]
    return line[0].split(":")[1]

def parse(args):
    datasets = ("Enron.g", "wikipedia-link-de.g", "trackers.g", "soc-Journal.g", "dblp-author.g", "patentcite.g", "soc-pokec-relationships.g", "wikiTalk.g")
    print(len(args), args)
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
        ds = arg 
    else:
        print("Please provide valid dataset: ", enumerate(datasets))
        exit(0)
    return ds

def runSim(datasets):
    results = []
    for ds in datasets:
        output = sp.run(["./kcore", ds], stdout=PIPE, stderr=PIPE)
        time = parseResult(str(output.stdout))
        results.append(tuple(ds, time))
    return results


if __name__ == "__main__":
    sp.run(["git", "pull"])
    sp.run(["make"])
    datasets = parse(sys.argv)
    results = runSim(datasets)
    for ds, time in results:
        print(ds, time)