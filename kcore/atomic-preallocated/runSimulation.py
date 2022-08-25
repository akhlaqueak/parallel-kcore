import os
import subprocess as sp
import sys

def parseResult(output):
    # One of the line in output has this format
    # Elapsed Time: 69.8055
    line = [x for x in output.split("\n") if "Elapsed Time" in x]
    return line[0].split(":")[1]

def parse(args):
    datasets = ("Enron.g", "wikipedia-link-de.g", "trackers.g", "soc-Journal.g", "dblp-author.g", "patentcite.g", "soc-pokec-relationships.g", "wikiTalk.g")
    arg = "" if len(args)<1 else args[1]

    if(len(args)<1):
        arg = ""
    # if arg == "":
    #     ds.append("Enron.g")
    if arg.isnumeric():
        ind = int(arg)
        if(ind<len(datasets)):
            ds.append(datasets[int(arg)])
        else:
            print("Please provide valid dataset: ", enumerate(datasets))
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
        output = sp.run(["./kcore", ds])
        time = parseResult(output.stdout)
        results.append(tuple(ds, time))
    return results


if __name__ == "__main__":
    sp.run(["git", "pull"])
    sp.run(["make"])
    datasets = parse(sys.argv)
    results = runSim(datasets)
    for ds, time in results.items():
        print(ds, time)