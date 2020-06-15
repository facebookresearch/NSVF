import sys, glob
import numpy as np

data_path = sys.argv[1]
np.savetxt(
    data_path + '/test_traj.txt',
    np.concatenate([np.loadtxt(f) for f in glob.glob(data_path + '/pose/*.txt')[200:]], 0)
)