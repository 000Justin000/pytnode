import sys 
import os

path = sys.argv[1]
suffix = sys.argv[2]
for filename in os.listdir(path):
    if filename[-4:] == ".png":
        itr, nid, tid = filename[:-4].split('_')
        if itr == suffix:
            os.system("convert -append" + "    " + os.path.join(path, "00000_"+nid+"_"+tid+".png") + "    " + os.path.join(path, filename) + "    " + os.path.join(path, "combine/"+nid+"_"+tid+".png"))
