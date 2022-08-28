import os
import time
import sys
import subprocess as sp

DSLOC = "./data_set/data/ours_format/"
REMOVE = True

def convert_format(ds):
    outfile = DSLOC + ds + ".g"
    zipfile = "download.tsv." + ds + ".tar.bz2"
    konnectloc = "http://konect.cc/files/" + zipfile
    tsvfile = ds + "/" + "out." + ds

    os.system("wget " + konnectloc)
    os.system("tar -xvf " + zipfile)


    comment_ch = '#@%!'
    st = time.time()
    lines = []
    out_lines = []
    max_node = 0

    with open(tsvfile,'r') as reader:
        lines = reader.readlines()

    for line in lines:
        if line[0] in comment_ch:
            continue
        s, e = line.split()[:2]
        if s == e:
            continue
        out_lines.append(s + " " + e)
        max_node = max(max_node, int(s), int(e))

    with open(outfile, 'w') as writer:
        writer.write(str(max_node+1)+'\n')
        writer.writelines(out_lines)

    if REMOVE:
        os.remove(zipfile)
        # os.removedirs(ds)

    print("Elapsed: ", time.time()-st)

if __name__ == "__main__":
    datasets = sys.argv[1:]
    for ds in datasets:
        convert_format(ds)