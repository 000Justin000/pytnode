import sys
import os

path = sys.argv[1]
for filename in os.listdir(path):
    if filename[-4:] == ".png":
        sid, nid, tid = filename[:-4].split('_')
        tid = tid.zfill(4)
        new_filename = sid + "_" + nid + "_" + tid + ".png"
        os.rename(os.path.join(path, filename), os.path.join(path, new_filename))
