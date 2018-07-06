#!/usr/bin/python
# Usage:
# python ./tools/generate_record_file.py ...

import os
import sys
import argparse
import subprocess

curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curr_path, "../"))

from dataset.pascal_voc import VOCDetection


def parse_args():
    parser = argparse.ArgumentParser(description='Generate record file for dataset')
    parser.add_argument('--dataset-name', type=str, default='voc_07_test',
                        help='dataset name')
    parser.add_argument('--dataset-path', type=str, default='./data/VOCdevkit',
                        help='dataset path')
    parser.add_argument('--target-path', type=str, default='./data/',
                        help='output list and record file directory')
    args = parser.parse_args()
    return args


def get_dataset(dataset_name, dataset_path):
    dataset_name = dataset_name.lower()
    if dataset_name == 'voc_0712_trainval':
        return VOCDetection(splits=[(2007, 'trainval'), (2012, 'trainval')], root=dataset_path, preload_label=False)
    elif dataset_name == 'voc_07_test':
        return VOCDetection(splits=[(2007, 'test')], root=dataset_path, preload_label=False)
    else:
        raise NotImplementedError('Dataset: {} not implemented.'.format(dataset_name))


if __name__ == '__main__':
    args = parse_args()

    print("Init {} dataset...".format(args.dataset_name))
    dataset = get_dataset(args.dataset_name, args.dataset_path)

    print("Read image list and annotations...")
    imglist = dataset.get_imglist()

    list_file = os.path.join(args.target_path, args.dataset_name + '.lst')
    print("Saving list file {} to disk...".format(list_file))
    with open(list_file, 'w') as f:
        for line in imglist:
            f.write(line)

    rec_file = os.path.join(args.target_path, args.dataset_name + '.rec')
    # Only shuffle in data loader
    print("Generating record file {}...".format(rec_file))
    subprocess.check_call(["python", "./tools/im2rec.py", list_file, './', "--no-shuffle",
                           "--pack-label", "--num-thread", "4"])

    print("All Done")
