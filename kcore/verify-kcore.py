import networkx as nx
import json
import sys
def verify(arg):
    dataset = arg[1]
    nx_kcore = {}
    try:
        file = open("./output/nx-kcore-" + dataset, 'r')
        nx_kcore = json.load(file)
        file.close()
    except IOError:
        G = nx.read_adjlist("./data_set/data/ours_format/" + dataset)
        nx_kcore = nx.core_number(G)
        # save the file for future use... 
        json.dump(nx_kcore, open("./output/nx-kcore-" + dataset, 'w'))

    pkc_kcore = json.load(open("./output/pkc-kcore-" + dataset, 'r'))

    if nx_kcore == pkc_kcore:
        print("Test Passed!")
    else:
        print("Test Failed!")
        print("The difference is: ")
        diff = set(nx_kcore.items()) ^ set(pkc_kcore.items())
        print (len(diff), " items")

if __name__ == "__main__":
    verify(sys.argv)
