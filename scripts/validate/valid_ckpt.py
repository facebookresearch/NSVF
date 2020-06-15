import sys, os, subprocess
import torch

MODEL_ROOT="/checkpoint/jgu/space/neuralrendering/debug_new_singlev3/"
MODEL_PATH, FACTOR, GAMMA = sys.argv[1:4]
CHECKPOINT_FILE = os.path.join(MODEL_ROOT, MODEL_PATH, 'checkpoint_last.pt')
config = torch.load(CHECKPOINT_FILE)['args']

DATA = config.data
try:
    VALIDVIEW = sys.argv[4]
except:
    VALIDVIEW = config.valid_views
RES = config.view_resolution

COMMAND = "bash scripts/validate/validate_nerf_test.sh {} {} {} {} {} {}".format(
    DATA, MODEL_PATH, RES, VALIDVIEW, FACTOR, GAMMA
)
print(COMMAND)
p = subprocess.Popen(COMMAND, stdout=subprocess.PIPE, shell=True)
while True:
    line = p.stdout.readline()
    if not line: break
    print(line)
