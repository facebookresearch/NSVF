import sys, os, subprocess

MODEL_ROOT="/checkpoint/jgu/space/neuralrendering/debug_new_singlev3/"
MODEL_PATH, FACTOR, GAMMA = sys.argv[1:4]
LOG_FILE = os.path.join(MODEL_ROOT, MODEL_PATH, 'train.log')
if os.path.exists(LOG_FILE):
    print(LOG_FILE)
    for line in open(LOG_FILE):
        if 'fairseq_cli.train | Namespace' in line:
            config = eval(line.split('|')[-1].replace('Namespace', 'dict'))
            DATA = config['data']
            VALIDVIEW = config['valid_views']
            RES = config['view_resolution']
            
            COMMAND = "bash scripts/validate/validate_nerf_test.sh {} {} {} {} {} {}".format(
                DATA, MODEL_PATH, RES, VALIDVIEW, FACTOR, GAMMA
            )

            p = subprocess.Popen(COMMAND, stdout=subprocess.PIPE, shell=True)
            while True:
                line = p.stdout.readline()
                if not line: break
                print(line)
            break

else:
    raise FileNotFoundError('no log found. need to launch manually')