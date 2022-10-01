import networkx as nx
import json
import time

datasets = [
'amazon0601.txt', \
'web-Google.txt', \
'dblp-author.txt', \
'patentcite.txt', \
'web-BerkStan.txt', \
'wiki-Talk.txt', \
'as-skitter.txt', \
'in-2004.txt', \
'wb-edu.txt', \
'soc-LiveJournal1.txt', \
'com-orkut.txt', \
'hollywood-2009.txt', \
'uk-2002.txt', \
'wikipedia-link-de.txt', \
'uk-2005.txt', \
'indochina-2004.txt', \
'trackers.txt', \
'arabic-2005.txt', \
'webbase-2001.txt', \
'it-2004.txt', \
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
    except Exception as e:
        print(str(e))
        print("Failed... ", ds)
    # json.dump(nxkcore, open("./output/networkx-kcore-" + ds, 'w'))
