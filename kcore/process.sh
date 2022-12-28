res=["Dataset"
"Ours"
"Ours + shared Mem"
"Ours + Vertex Prefetching:"
"Efficient scan:"
"Ballot scan:"
"Efficient scan + Shared Memory + :"
"Ballot scan + Shared Memory:"
"Efficient Scan + Vertex Prefetching:"
"Ballot scan + Vertex Prefetching:"
]

for ds in $res; do
    echo "$ds\n\n\n"
    grep "$ds" log/publish.out | awk '{print $NF}'
done