import networkx as nx
import json
# ds = ['Enron.g','gowalla.g','roadNetCa.g','roadNetPa.g','roadNetTx.g','wikiTalk.g']
ds = ['33.g']

for dataset in ds:
    G = nx.read_adjlist("./data_set/data/ours_format/" + dataset)
    nxkcore = nx.core_number(G)
    json.dump(nxkcore, open("output/" + dataset + "-nx-kcore", 'w'))
    # with open("output/" + dataset + "-nx-kcore", 'w') as f: 
    #     f.write(json.dumps(nxkcore))
        # for key, value in details.items(): 
        #     f.write('%s %s\n' % (key, value))