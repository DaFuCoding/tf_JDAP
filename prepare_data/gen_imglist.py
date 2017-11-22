#!/usr/bin/env python

import numpy as np
import numpy.random as npr
import os

save_data_dir = '/home/dafu/data/jdap_data'

netSize = 12

dir_path = os.path.join(save_data_dir, '%s' % netSize)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

with open(os.path.join(save_data_dir, '%s/pos_%s.txt' % (netSize, netSize)), 'r') as f:
    pos = f.readlines()

with open(os.path.join(save_data_dir, '%s/neg_%s.txt' % (netSize, netSize)), 'r') as f:
    neg = f.readlines()

with open(os.path.join(save_data_dir, '%s/part_%s.txt' % (netSize, netSize)), 'r') as f:
    part = f.readlines()

with open(os.path.join(dir_path, "train_%s.txt" % netSize), "w") as f:
    nums = [len(neg), len(pos), len(part)]
    ratio = [1, 1.5, 3]
    base_num = len(pos)
    print(len(pos), len(part), len(neg), base_num)
    pos_keep = npr.choice(len(pos), size=base_num * ratio[0], replace=False)
    if base_num * ratio[1] > len(part):
        part_keep = npr.choice(len(part), size=int(len(part)), replace=False)
    else:
        part_keep = npr.choice(len(part), size=int(base_num * ratio[1]), replace=False)
    if base_num * ratio[2] > len(neg):
        neg_keep = npr.choice(len(neg), size=int(len(neg)), replace=False)
    else:
        neg_keep = npr.choice(len(neg), size=base_num * ratio[2], replace=False)

    for i in pos_keep:
        f.write(pos[i])
    for i in neg_keep:
        f.write(neg[i])
    for i in part_keep:
        f.write(part[i])

with open(os.path.join(save_data_dir, '%s/val_pos_%s.txt' % (netSize, netSize)), 'r') as f:
    val_pos = f.readlines()

with open(os.path.join(save_data_dir, '%s/val_neg_%s.txt' % (netSize, netSize)), 'r') as f:
    val_neg = f.readlines()

with open(os.path.join(dir_path, "val_%s.txt" % netSize), "w") as f:
    f.writelines(val_neg)
    f.writelines(val_pos)