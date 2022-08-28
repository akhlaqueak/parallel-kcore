import networkx as nx
import json
import time
# datasets = ["soc-Journal.g", "dblp-author.g", "patentcite.g", "soc-pokec-relationships.g"]
# datasets = ["wikipedia-link-de.g"]
datasets = ["Enron.g", "wikiTalk.g"]
# datasets = ['patentcite.g']
for ds in datasets:
    tick = time.time()
    print(ds, " loading started at ", time.ctime())
    G = nx.read_edgelist("./data_set/data/ours_format/" + ds)
    tock = time.time()
    print(ds, " loading completed in ", tock-tick)
    
    # tick = time.time()
    # G = nx.read_adjlist("./data_set/data/ours_format/" + ds)
    # tock = time.time()
    # print("Time in edgelist", tock-tick)
    
    
    tick = time.time()
    print(ds, " processing started at ", time.ctime())
    nxkcore = nx.core_number(G)
    tock = time.time()
    print(ds, " processing completed in ", tock-tick)

    tick = time.time()
    print(ds, " writing started at ", time.ctime())
    json.dump(nxkcore, open("./output/gennx-kcore-" + ds, 'w'))
    tock = time.time()
    print(ds, " writing completed in ", tock-tick)
