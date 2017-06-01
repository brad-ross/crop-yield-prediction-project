import os
import numpy as np
import sys

listpath = sys.argv[1]
outpath = sys.argv[2]

filenames = os.listdir(os.path.expanduser(listpath))
with open(os.path.expanduser(outpath), "w") as f:
    for filename in filenames:
        f.write(str(filename) + "\n")
