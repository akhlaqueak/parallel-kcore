import networkx as nx
import json
import time

datasets = [
'amazon0601.txt', \
'wiki-Talk.txt', \
'web-Google.txt', \
'web-BerkStan.txt', \
'as-skitter.txt', \
'patentcite.txt', \
'in-2004.txt', \
'dblp-author.txt', \
# 'wb-edu.txt', \
# 'soc-LiveJournal1.txt', \
# 'wikipedia-link-de.txt', \
# 'hollywood-2009.txt', \
# 'com-orkut.txt', \
# 'trackers.txt', \
# 'indochina-2004.txt', \
# 'uk-2002.txt', \
# 'arabic-2005.txt', \
# 'uk-2005.txt', \
# 'webbase-2001.txt', \
# 'it-2004.txt', \
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
        now = time.strftime("%H:%M", time.localtime(time.time()))
        print(str(e))
        print(now, "Failed... ", ds)
    # json.dump(nxkcore, open("./output/networkx-kcore-" + ds, 'w'))
