import os
import argparse
import subprocess
from .dataset.pascal_voc import PascalVOC
from .dataset.mscoco import Coco
from .dataset.concat_db import ConcatDB

curr_path = os.path.abspath(os.path.dirname(__file__))


def load_pascal_voc(image_set, year, root_path, shuffle=False, **kwargs):
    """
    wrapper function for loading pascal voc dataset

    Parameters:
    ----------
    image_set : str
        train, trainval...
    year : str
        2007, 2012 or combinations splitted by comma
    devkit_path : str
        root directory of dataset
    shuffle : bool
        whether to shuffle initial list

    Returns:
    ----------
    Imdb
    """
    image_set = [y.strip() for y in image_set.split(',')]
    assert image_set, "No image_set specified"
    year = [y.strip() for y in year.split(',')]
    assert year, "No year specified"

    # make sure (# sets == # years)
    if len(image_set) > 1 and len(year) == 1:
        year = year * len(image_set)
    if len(image_set) == 1 and len(year) > 1:
        image_set = image_set * len(year)
    assert len(image_set) == len(year), "Number of sets and year mismatch"

    imdbs = []
    for s, y in zip(image_set, year):
        imdbs.append(PascalVOC(s, y, root_path, shuffle, is_train=True))
    if len(imdbs) > 1:
        return ConcatDB(imdbs, shuffle)
    else:
        return imdbs[0]


def load_coco(image_set, root_path, shuffle=False, **kwargs):
    """
    wrapper function for loading ms coco dataset

    Parameters:
    ----------
    image_set : str
        train2014, val2014, valminusminival2014, minival2014
    dirname: str
        root dir for coco
    shuffle: boolean
        initial shuffle
    """
    anno_files = ['instances_' + y.strip() + '.json' for y in image_set.split(',')]
    assert anno_files, "No image set specified"
    imdbs = []
    for af in anno_files:
        af_path = os.path.join(root_path, 'annotations', af)
        imdbs.append(Coco(af_path, root_path, shuffle=shuffle))
    if len(imdbs) > 1:
        return ConcatDB(imdbs, shuffle)
    else:
        return imdbs[0]


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare lists for dataset')
    parser.add_argument('--dataset', dest='dataset', help='dataset to use',
                        default='pascal', type=str)
    parser.add_argument('--year', dest='year', help='which year to use',
                        default='2007,2012', type=str)
    parser.add_argument('--set', dest='set', help='train, val, trainval, test',
                        default='trainval', type=str)
    parser.add_argument('--target', dest='target', help='output list file',
                        default=os.path.join(curr_path, '..', 'train.lst'),
                        type=str)
    parser.add_argument('--root', dest='root_path', help='dataset root path',
                        default=os.path.join(curr_path, '..', 'data', 'VOCdevkit'),
                        type=str)
    parser.add_argument('--shuffle', dest='shuffle', help='shuffle list',
                        type=bool, default=True)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    datasets = {
        'voc': load_pascal_voc,
        'coco': load_coco
    }
    dataset = args.dataset.lower()
    if dataset not in datasets:
        raise ValueError('Dataset %s is not supported. Available options are\n\t%s' % (
                dataset, '\n\t'.join(sorted(datasets.keys()))))
    print("Read image list and annotations...")
    db = datasets[dataset](image_set=args.set, year=args.year, root_path=args.root_path, shuffle=args.shuffle)
    print("saving list to disk...")
    db.save_imglist(args.target, root=args.root_path)

    print("List file {} generated...".format(args.target))

    # Only shuffle in generating list
    subprocess.check_call(["python",
        os.path.join(curr_path, "im2rec.py"),
        os.path.abspath(args.target), os.path.abspath(args.root_path),
        "--no-shuffle", "--pack-label", "--num-thread", "4"])

    print("Record file {} generated...".format(args.target.rsplit('.')[0] + '.rec'))
