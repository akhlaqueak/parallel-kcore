import networkx as nx
import json
import sys
def verify(file):
    dataset = file[1]

    nx_kcore = json.load(open("output/" + dataset + "-nx-kcore", 'r'))
    pkc_kcore = json.load(open("output/" + dataset + "-pkc-kcore", 'r'))

    if nx_kcore == pkc_kcore:
        print("Test Passed!")
    else:
        print("Test Failed!")
        diff = set(nx_kcore.items()) ^ set(pkc_kcore.items())
        print (diff)

if __name__ == "__main__":
    verify(sys.argv)
