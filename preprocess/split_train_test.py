# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse, random, glob, os


parser = argparse.ArgumentParser(description='Split the train/test sets given an offline dataset')
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--train_ratio', type=float, default=0.9)

args = parser.parse_args()

ftrain = open(os.path.join(args.output_dir, 'train.txt'), 'w')
ftest  = open(os.path.join(args.output_dir, 'test.txt'), 'w')

file_lists = glob.glob(args.data_dir + '/*/models/model_normalized.obj')
random.shuffle(file_lists)

for i, file in enumerate(file_lists):
    filename = '/'.join(file.split('/')[:-2])
    if i / len(file_lists) < args.train_ratio:
        print(filename, file=ftrain)
    else:
        print(filename, file=ftest)
