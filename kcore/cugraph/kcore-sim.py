import time
import cudf
import cugraph

datasets = ['amazon0601',
            "wiki-Talk",
            "web-Google",
            "web-BerkStan",
            "as-skitter",
            "patentcite",
            "in-2004",
            "dblp-author",
            "wb-edu",
            "soc-LiveJournal1",
            "wikipedia-link-de",
            "hollywood-2009",
            "com-orkut",
            "trackers",
            "indochina-2004",
            "uk-2002",
            "arabic-2005",
            "uk-2005",
            "webbase-2001",
            "it-2004"
            ]
for ds in datasets:
    dstxt = "DataBank/" + ds+".txt"
    try:
        print("File loading : ", dstxt)
        df = cudf.read_csv(dstxt, comment='#', sep='\t',
                        skipinitialspace=True, header=None)
        if df.shape[1] < 2:
            df = cudf.read_csv(dstxt, comment='#', sep=' ',
                            skipinitialspace=True, header=None)
        G = cugraph.Graph()
        print("Creating graph : ", dstxt)
        G.from_cudf_edgelist(df, source='0', destination='1', renumber=False)
        print("Calculating k-core : ", dstxt)
        tick = time.time()
        core = cugraph.core_number(G)
        tock = time.time()
        print("k-core calculation time: ", dstxt, tock-tick)
    except:
        print("Unsuccessful: ", dstxt)
