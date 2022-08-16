import codecs
import os
import time
import sys
def convert_format(arg):
    in_file = arg[1]
    graph = in_file.split('.')[0] + ".g"

    comment_ch = '#@%!'
    st = time.time()
    lines = []
    out_lines = []
    max_node = 0

    with codecs.open(in_file,'r', 'utf-8') as reader:
        lines = reader.readlines()

    for line in lines:
        if line[0] in comment_ch:
            continue
        s, e = line.split()
        if s == e:
            continue
        out_lines.append(line)
        max_node = max(max_node, int(s), int(e))

    with codecs.open(graph, 'w', 'utf-8') as writer:
        writer.write(str(max_node+1)+'\n')
        writer.writelines(out_lines)

    print("Elapsed: ", time.time()-st)

if __name__ == "__main__":
    convert_format(sys.argv)