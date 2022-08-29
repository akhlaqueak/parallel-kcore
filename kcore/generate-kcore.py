import networkx as nx
import json
import time
# datasets = ["soc-Journal.g", "dblp-author.g", "patentcite.g", "soc-pokec-relationships.g"]
datasets = ["wikipedia-link-de.g"]
# datasets = ['patentcite.g']
for ds in datasets:
    tick = time.time()
    G = nx.read_edgelist("./data_set/data/ours_format/" + ds)
    tock = time.time()
    print("Time in edgelist", tock-tick)
    
    
    tick = time.time()
    nxkcore = nx.core_number(G)
    tock = time.time()
    print(ds, tock-tick)
    json.dump(nxkcore, open("./output/gennx-kcore-" + ds, 'w'))
