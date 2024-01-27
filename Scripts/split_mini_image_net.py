import os
import shutil
import random
import torch
import argparse


"""
All 100 classes must be in the "all" sub-directoy.
The program will split the images to the train and test sub-directories. Please not that it removes those directories if they exist.

An example: If you use "-n 500" for Mini-ImageNet, the program puts 500
samples in the train set and the remaining 100 samples in the test set.
"""


def list_directories_and_files(directory):
    directories = []
    for file in os.listdir(directory):
        directories.append(file)
    return directories


def set_seed(seed: int):
    random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def split(dir_all, dir_train, dir_test, num_train_samples):
    dirs = list_directories_and_files(dir_all)
    random.shuffle(dirs)
    
    for dir in dirs:
        files_list = list_directories_and_files(os.path.join(dir_all, dir))
        random.shuffle(files_list)
        assert len(files_list) == 600
        cnt = 0

        dest_dir_train = os.path.join(dir_train, dir)
        dest_dir_test = os.path.join(dir_test, dir)
        os.makedirs(dest_dir_train, exist_ok=True)
        os.makedirs(dest_dir_test, exist_ok=True)
        
        for f in files_list:
            source = os.path.join(dir_all, dir, f)
            assert os.path.isfile(source)
            if cnt < num_train_samples:
                dest = os.path.join(dest_dir_train, f)
            else:
                dest = os.path.join(dest_dir_test, f)
            shutil.copy(source, dest)
            cnt += 1


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-root', type=str, default='', required=True,
                        help='root directory')
    
    parser.add_argument('-n', type=int, default=500, required=False,
                        help='number of train samples')
    
    parser.add_argument('-seed', type=int, default=0, required=False,
                        help='seed number')
    
    return parser.parse_args()


def main():
    args = parse_arguments()

    root_dir = args.root
    num_train_samples = int(args.n)
    set_seed(int(args.seed))
    dir_all = os.path.join(root_dir, 'all')
    dir_train = os.path.join(root_dir, 'train')
    dir_test = os.path.join(root_dir, 'test')
    if os.path.exists(dir_train):
        shutil.rmtree(dir_train)
    if os.path.exists(dir_test):
        shutil.rmtree(dir_test)
    os.makedirs(dir_train, exist_ok=True)
    os.makedirs(dir_test, exist_ok=True)

    split(dir_all, dir_train, dir_test, num_train_samples)


if __name__ == '__main__':
    main()
