import cudf, cugraph
ds = 'DataBank/trackers.txt'
df = cudf.read_csv(ds, comment='#', sep='\t', skipinitialspace=True, header=None)

if df.shape[1]<2 :
    df = cudf.read_csv(ds, comment='#', sep=' ', skipinitialspace=True, header=None)

G = cugraph.Graph()
G.from_cudf_edgelist(df, source='0', destination='1', renumber=True)
core = cugraph.core_number(G)