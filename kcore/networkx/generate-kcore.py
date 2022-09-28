import networkx as nx
import json
import time
# datasets = ["soc-Journal.g", "dblp-author.g", "patentcite.g", "soc-pokec-relationships.g"]
datasets = ['web-BerkStan.txt',\
'soc-LiveJournal1.txt',\
'wb-edu.txt',\
'hollywood-2009.txt',\
'web-Google.txt',\
'wiki-Talk.txt',\
]
# datasets = ['patentcite.g']
for ds in datasets:
    tick = time.time()
    G = nx.read_edgelist("./data_set/data/" + ds)
    tock = time.time()
    print("Time in loading", ds, tock-tick)
    
    
    tick = time.time()
    nxkcore = nx.core_number(G)
    tock = time.time()
    print("Excecution time: ", ds, tock-tick)
    # json.dump(nxkcore, open("./output/networkx-kcore-" + ds, 'w'))
