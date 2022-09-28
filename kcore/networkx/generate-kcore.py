import networkx as nx
import json
import time
# datasets = ["soc-Journal.g", "dblp-author.g", "patentcite.g", "soc-pokec-relationships.g"]
datasets = ['amazon0601.txt', \
'web-BerkStan.txt', \
'web-Google.txt', \
'hollywood-2009.txt', \
'in-2004.txt', \
'as-Skitter .txt', \
'wiki-Talk.txt', \
'wikipedia-link-de.txt', \
'patentcite.txt', \
'soc-LiveJournal1.txt', \
'dblp-author.txt', \
'indochina-2004.txt', \
'wb-edu.txt', \
'uk-2002.txt', \
'arabic-2005.txt', \
]
# datasets = ['patentcite.g']
for ds in datasets:
    print("Starting time", ds, time.time())
    tick = time.time()
    G = nx.read_edgelist("../data_set/data/" + ds)
    tock = time.time()
    print("Time in loading", ds, tock-tick)
    
    
    tick = time.time()
    nxkcore = nx.core_number(G)
    tock = time.time()
    print("Excecution time: ", ds, tock-tick)
    # json.dump(nxkcore, open("./output/networkx-kcore-" + ds, 'w'))
