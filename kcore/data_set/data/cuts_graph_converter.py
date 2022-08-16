import codecs
import os
import time
import sys
def convert_format(arg):
    snap_graphs = arg[1]
    cuts_graphs = snap_graphs.split('.')[0] + ".g"
    comment_ch = '#@%!'
    st = time.time()
    for i, graph in enumerate(snap_graphs):
        lines = []
        out_lines = []
        max_node = 0

        writter = open(cuts_graphs[i], 'w')
        with codecs.open(graph,'r','utf-8') as reader:
            lines = reader.readlines()
        for line in lines:
            if line[0] in comment_ch:
                continue
            s, e = line.split()
            if s == e:
                continue
            out_lines.append(line)
            max_node = max(max_node, int(s), int(e))
        writter.write(str(max_node+1)+'\n')
        writter.writelines(out_lines)
        writter.close()
    print("Elapsed: ", time.time()-st)

if __name__ == "__main__":
    convert_format(sys.argv)