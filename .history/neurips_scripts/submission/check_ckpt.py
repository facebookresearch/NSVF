import sys, os, subprocess
import torch

MODEL_ROOT="/checkpoint/jgu/space/neuralrendering/debug_new_singlev3/"
MODEL_PATH = sys.argv[1]
CHECKPOINT_FILE = os.path.join(MODEL_ROOT, MODEL_PATH, 'checkpoint_last.pt')
config = torch.load(CHECKPOINT_FILE)['args']

print(config.data)
print(config.view_resolution)
