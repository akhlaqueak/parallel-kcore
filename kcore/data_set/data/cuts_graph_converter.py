import os
import time
import sys

# This file converts the graphs available at 
# 1. konect.cc
# 2. stanford snap
# 3. https://sparse.tamu.edu/

# it removes cycles in the graph, and create the graph in 
# (first line is number of vertices, followed by edge at every line)

# V
# source destination
# source destination
# source destination
# source destination
# source destination

def convert_format(arg):
    in_file = arg[1]
    graph = in_file.split('.')[0] + ".g"

    comment_ch = '#@%!'
    st = time.time()
    lines = []
    out_lines = []
    max_node = 0

    with open(in_file,'r') as reader:
        lines = reader.readlines()

    for line in lines:
        if line[0] in comment_ch:
            continue
        s, e = line.split()
        if s == e: # it's a self-loop.
            continue
        out_lines.append(line)
        max_node = max(max_node, int(s), int(e))

    with open(graph, 'w') as writer:
        writer.write(str(max_node+1)+'\n')
        writer.writelines(out_lines)

    print("Elapsed: ", time.time()-st)

if __name__ == "__main__":
    convert_format(sys.argv)