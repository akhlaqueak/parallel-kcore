import networkx as nx
dataset = "wikiTalk.g"
G = nx.read_adjlist("../data_set/data/ours_format/" + dataset)
nxkcore = nx.core_number(G)
d = {}
with open("../data_set/data/ours_format/" + dataset + "-kcore.txt") as f:
    d[f.readline().strip()] = 0
    for line in f:
        (key, val) = line.split()
        if val!= '0':
            d[key] = int(val)
if d == nxkcore:
    print("Test Passed!")
else:
    print("Test Failed!")