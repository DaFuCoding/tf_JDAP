#!/usr/bin/env python

import numpy as np
import numpy.random as npr
import os

save_data_dir = '/home/dafu/data/jdap_data'

netSize = 48
add_dir_name = 'mnet_'
mode = 'train'
dir_path = os.path.join(save_data_dir, '%d' % netSize)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
if mode == 'train':
    with open(os.path.join(save_data_dir, '%d/%s_%spos_%d.txt' %
            (netSize, mode, add_dir_name, netSize)), 'r') as f:
        pos = f.readlines()

    with open(os.path.join(save_data_dir, '%d/%s_%sneg_%d.txt' %
            (netSize, mode, add_dir_name, netSize)), 'r') as f:
        neg = f.readlines()

    with open(os.path.join(save_data_dir, '%d/%s_%spart_%d.txt' %
            (netSize, mode, add_dir_name, netSize)), 'r') as f:
        part = f.readlines()

    with open(os.path.join(dir_path, "%s_%s%d.txt" % (mode, add_dir_name, netSize)), "w") as f:
        nums = [len(neg), len(pos), len(part)]
        ratio = [1, 1.3, 3]
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

elif mode == 'val':
    with open(os.path.join(save_data_dir, '%d/%s_%spos_%s.txt' %
            (netSize, mode, add_dir_name, netSize)), 'r') as f:
        val_pos = f.readlines()
        base_num = len(val_pos)

    with open(os.path.join(save_data_dir, '%d/%s_%sneg_%s.txt' %
            (netSize, mode, add_dir_name, netSize)), 'r') as f:
        val_neg = f.readlines()
        neg_keep = npr.choice(len(val_neg), size=base_num * 3, replace=False)

    with open(os.path.join(dir_path, "%s_%s%d.txt" %
            (mode, add_dir_name, netSize)), "w") as f:
        f.writelines(val_pos)
        for i in neg_keep:
            f.write(val_neg[i])
