import os
import os.path as osp

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from configs.Configurations import Configurations
# import random


class MiniImageNet(Dataset):
    def __init__(self,
                 configs: Configurations,
                 train: bool,
                 dino: bool,
                 selected_classes: torch.Tensor,
                 task_id: int):
        
        self.configs = configs
        self.dino = dino
        self._train_mode = train    # training set or test set
        
        if train:
            split_file_name = 'train'
        else:
            split_file_name = 'test'

        self.selected_classes: list = selected_classes.tolist()
        root = configs.configs_dataset.dataroot
        self.root = os.path.expanduser(root)
        self.IMAGE_PATH = os.path.join(root, 'miniimagenet', 'images')
        self.SPLIT_PATH = os.path.join(root, 'miniimagenet', 'split')

        csv_path = osp.join(self.SPLIT_PATH, split_file_name + '.csv')
        lines_all_CSV_split = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        self.label_translation_list = None
        self.paths_samples_list = None
        self.labels_real_list = None
        self.image_name_to_real_label_dict = None
        self.label_names_all = None
        self.file_names_list = None
        self.prepare_required_lists_and_dictionaries(lines_all_CSV_split)

        self.transform_train = None
        self.transform_test = None
        self.set_transform_based_on_training_mode()

        num_shots = self.configs.configs_dataset.num_shots if train and task_id > 0 else -1
        
        if train:
            file_name_prefix = 'train'
        else:
            file_name_prefix = 'test'

        file_name = file_name_prefix + f'_samples_for_task={task_id + 1}.txt'

        self.path_text_file_for_this_task = os.path.join(self.configs.directory_permutation_files, file_name)

        if not os.path.exists(self.path_text_file_for_this_task):
            self.create_session_file(num_shots)

        self.use_samples_in_the_text_file()

    def set_transform_based_on_training_mode(self):
        image_size = self.configs.image_size

        self.transform_train = None
        self.transform_test = None

        if self.dino:
            self.transform_train = self.configs.dino_transforms
        else:
            self.transform_train = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        
        self.transform_test = transforms.Compose([
            # This following line is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate
            transforms.Resize([int(image_size * 1.15), int(image_size * 1.15)]),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])

        if self._train_mode:
            self.transform = self.transform_train
        else:
            self.transform = self.transform_test

    def prepare_required_lists_and_dictionaries(self, lines_all_CSV_split: str):
        paths_samples_list = []
        labels_real_list = []
        file_names_list = []
        image_path_to_real_label = {}
        label_index = -1

        label_names_all = []

        for line in lines_all_CSV_split:
            file_name, label_real_name = line.split(',')
            path_image = osp.join(self.IMAGE_PATH, file_name)
            if label_real_name not in label_names_all:  # A new class
                label_names_all.append(label_real_name)
                label_index += 1
            file_names_list.append(file_name)
            paths_samples_list.append(path_image)
            labels_real_list.append(label_index)
            image_path_to_real_label[file_name] = label_index

        # Translating labels to begin from zero index
        label_translation = [0] * len(self.configs.class_permutation)
        index = 0
        for i in self.configs.class_permutation.tolist():
            label_translation[i] = index
            index += 1

        self.label_translation_list = label_translation
        self.paths_samples_list = paths_samples_list
        self.labels_real_list = labels_real_list
        self.image_name_to_real_label_dict = image_path_to_real_label
        self.label_names_all = label_names_all
        self.file_names_list = file_names_list

    def use_samples_in_the_text_file(self) -> tuple[list, list]:
        file_names_list = [x.strip() for x in open(self.path_text_file_for_this_task, 'r').readlines()]
        
        paths_samples_list_new = []
        labels_list_new = []
        
        for file_name in file_names_list:
            path_image = os.path.join(self.IMAGE_PATH, file_name)
            paths_samples_list_new.append(path_image)
            label = self.image_name_to_real_label_dict[file_name]
            assert label in self.selected_classes   # We verify if the sample belongs to the current task.
            labels_list_new.append(label)

        self.paths_samples_list = paths_samples_list_new
        self.labels_real_list = labels_list_new
        self.configs.logger.info(f"The session file \"{self.path_text_file_for_this_task}\" is loaded!")

    def create_session_file(self, num_shots=-1):
        self.configs.set_seed()
        samples_list_selected = []

        for label in self.selected_classes:
            selected_indices = np.where(label == np.array(self.labels_real_list))[0]

            if num_shots > 0:
                np.random.shuffle(selected_indices)
                selected_indices = selected_indices[:num_shots]
                assert len(selected_indices) == num_shots

            for j in selected_indices:
                samples_list_selected.append(self.file_names_list[j])

        with open(self.path_text_file_for_this_task, "w") as text_file:
            for file_name in samples_list_selected:
                text_file.write(file_name + '\n')
        text_file.close()

        self.configs.logger.info(f"The session file \"{self.path_text_file_for_this_task}\" is created for the first time!")

    def __len__(self):
        return len(self.paths_samples_list)

    def __getitem__(self, index):
        label_real = self.labels_real_list[index]
        labels_translated = self.label_translation_list[label_real]
        path = self.paths_samples_list[index]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, labels_translated, label_real


if __name__ == '__main__':
    txt_path = "../../data/index_list/mini_imagenet/session_1.txt"
    # class_index = open(txt_path).read().splitlines()
    base_class = 100
    class_index = np.arange(base_class)
    dataroot = '~/data'
    batch_size_base = 400
    trainset = MiniImageNet(root=dataroot, train=True, transform=None, task_id=0)
    cls = np.unique(trainset.labels_real_list)
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_base, shuffle=True, num_workers=8,
                                              pin_memory=True)
    print(trainloader.dataset.data.shape)
