import networkx as nx
import json
import time
datasets = ["soc-Journal.g", "dblp-author.g", "patentcite.g", "soc-pokec-relationships.g"]
# ds = ['33.g']

for ds in datasets:
    G = nx.read_adjlist("./data_set/data/ours_format/" + ds)
    tick = time.time()
    nxkcore = nx.core_number(G)
    tock = time.time()
    print(ds, tock-tick)
    json.dump(nxkcore, open("./output/gennx-kcore-" + ds, 'w'))
    # with open("output/" + dataset + "-nx-kcore", 'w') as f: 
    #     f.write(json.dumps(nxkcore))
        # for key, value in details.items(): 
        #     f.write('%s %s\n' % (key, value))