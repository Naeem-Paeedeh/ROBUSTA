import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from configs.Configurations import Configurations


class CUB200(Dataset):
    def __init__(self,
                 configs: Configurations,
                 train: bool,
                 dino: bool,
                 selected_classes: torch.Tensor,
                 task_id: int):
        self.configs = configs
        self.dino = dino
        self._train_mode = train  # training set or test set
        self.selected_classes: list = selected_classes.tolist()

        self.root = os.path.expanduser(configs.configs_dataset.dataroot)
        
        self.image_path_to_label_dict = {}
        self.label_translation_dict = {}
        self.prepare_required_lists_and_dictionaries()

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
            transforms.Resize([int(image_size * 1.15), int(image_size * 1.15)]),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])

        if self._train_mode:
            self.transform = self.transform_train
        else:
            self.transform = self.transform_test

    def prepare_required_lists_and_dictionaries(self):
        path_image_file = os.path.join(self.root, 'CUB_200_2011/images.txt')
        path_split_file = os.path.join(self.root, 'CUB_200_2011/train_test_split.txt')
        path_labels_file = os.path.join(self.root, 'CUB_200_2011/image_class_labels.txt')

        def list_to_dictionary(lines_list: list):
            dict = {}
            for line in lines_list:
                s = line.split(' ')
                id = int(s[0])
                cls = s[1]
                if id not in dict.keys():
                    dict[id] = cls
                else:
                    raise EOFError('The same ID can only appear once')
            return dict
        
        def text_read(file):
            with open(file, 'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    lines[i] = line.strip('\n')
            return lines

        id2image = list_to_dictionary(text_read(path_image_file))
        id2train = list_to_dictionary(text_read(path_split_file))  # 1: train images; 0: test iamges
        id2class = list_to_dictionary(text_read(path_labels_file))
        
        indices_list = []
        
        for k in sorted(id2train.keys()):
            belongs_to_train_set = id2train[k] == '1'
            if (belongs_to_train_set and self._train_mode) or (not (belongs_to_train_set or self._train_mode)):     # XNOR
                indices_list.append(k)
            
        self.paths_samples_list = []
        self.labels_real_list = []
        self.image_path_to_label_dict = {}
        for k in indices_list:
            image_relative_path = os.path.join('CUB_200_2011/images', id2image[k])
            self.paths_samples_list.append(image_relative_path)
            label = int(id2class[k]) - 1
            self.labels_real_list.append(label)
            self.image_path_to_label_dict[image_relative_path] = label

        # Translating labels to begin from zero index
        self.label_translation_dict = {}
        index = 0
        for i in self.configs.class_permutation.tolist():
            self.label_translation_dict[i] = index
            index += 1

    def use_samples_in_the_text_file(self):
        paths_list = open(self.path_text_file_for_this_task).read().splitlines()
        self.paths_samples_list = []
        self.labels_real_list = []
        
        for full_path_to_an_image in paths_list:
            self.paths_samples_list.append(full_path_to_an_image)
            label = self.image_path_to_label_dict[full_path_to_an_image]
            assert label in self.selected_classes   # We verify if the sample belongs to the current task.
            self.labels_real_list.append(label)

        self.configs.logger.info(f"The session file \"{self.path_text_file_for_this_task}\" is loaded!")

    def create_session_file(self, num_shots=-1):
        self.configs.set_seed()
        selected_paths_list = []
        for label in self.selected_classes:
            selected_indices = np.where(label == np.array(self.labels_real_list))[0]
            if num_shots > 0:
                np.random.shuffle(selected_indices)
                selected_indices = selected_indices[:num_shots]
                assert len(selected_indices) == num_shots
            
            for ind in selected_indices:
                selected_paths_list.append(self.paths_samples_list[ind])

        with open(self.path_text_file_for_this_task, "w") as text_file:
            for path in selected_paths_list:
                text_file.write(path + '\n')
        text_file.close()

        self.configs.logger.info(f"The session file \"{self.path_text_file_for_this_task}\" is created for the first time!")

    def __len__(self):
        return len(self.paths_samples_list)

    def __getitem__(self, i):
        label_real = self.labels_real_list[i]
        labels_translated = self.label_translation_dict[label_real]
        path = os.path.join(self.root, self.paths_samples_list[i])
        image = self.transform(Image.open(path).convert('RGB'))
        return image, labels_translated, label_real


if __name__ == '__main__':
    txt_path = "../../data/index_list/cub200/session_1.txt"
    # class_index = open(txt_path).read().splitlines()
    base_class = 100
    class_index = np.arange(base_class)
    dataroot = '~/dataloader/data'
    batch_size_base = 400
    trainset = CUB200(root=dataroot, train=False, selected_classes=class_index, base_session=True)
    cls = np.unique(trainset.labels_real_list)
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_base, shuffle=True, num_workers=8, pin_memory=True)
