import networkx as nx
import json
import time

datasets = [
'amazon0601', \
'web-Google', \
'dblp-author', \
'patentcite', \
'web-BerkStan', \
'wiki-Talk', \
'as-skitter', \
'in-2004', \
'wb-edu', \
'soc-LiveJournal1', \
'com-orkut', \
'hollywood-2009', \
'uk-2002', \
'wikipedia-link-de', \
'uk-2005', \
'indochina-2004', \
'trackers', \
'arabic-2005', \
'webbase-2001', \
'it-2004', \
]
# datasets = ['patentcite.g']
for ds in datasets:
    try:
        now = time.strftime("%H:%M", time.localtime(time.time()))
        print("Starting time", ds, now)
        tick = time.time()
        G = nx.read_edgelist("../data_set/data/" + ds)
        tock = time.time()
        print("Time in loading", ds, tock-tick)
        
        
        tick = time.time()
        nxkcore = nx.core_number(G)
        tock = time.time()
        print("Excecution time: ", ds, tock-tick)
    except:
        print("Failed... ", ds)
    # json.dump(nxkcore, open("./output/networkx-kcore-" + ds, 'w'))
