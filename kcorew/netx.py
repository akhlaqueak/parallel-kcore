import networkx as nx
import time
import json

DATA = "./data_set/data/ours_format/"
OUTPUT = "./output/"
datasets = ("Enron.g", "wikipedia-link-de.g", "trackers.g", "soc-Journal.g", "dblp-author.g", "patentcite.g", "soc-pokec-relationships.g", "twitter_mpi.g", "wikiTalk.g", "twitter.g")
for ds in datasets:
    G = nx.read_adjlist(DATA + ds)

    tick = time.time()
    kcore = nx.core_number(G)
    tock = time.time()

    print(ds, tock-tick)

    json.dump(kcore, open(OUTPUT + "nx-" + ds, 'w'))