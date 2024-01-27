import torch
from PIL import Image
import os
import os.path
import numpy as np
import pickle
import torchvision.transforms as transforms
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from configs.Configurations import Configurations


class CIFAR100(VisionDataset):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    # """

    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }

    def __init__(self,
                 configs: Configurations,
                 train: bool,
                 dino: bool,
                 selected_classes: torch.Tensor,
                 task_id: int,
                 target_transform=None,
                 download=False):
        self.configs = configs
        self.dino = dino
        self.task_id = task_id  # To be able to use a different task_id than the task_id in the configs
        self._train_mode = train  # training set or test set
        root = configs.configs_dataset.dataroot
        super().__init__(root, transform=None, target_transform=target_transform)
        self.root = os.path.expanduser(root)
        self.selected_classes: list = selected_classes.tolist()

        self.label_translation_list = None

        self.prepare_required_lists_and_dictionaries()

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted. You can use download=True to download it')

        self.transform_train = None

        self.transform_train = None
        self.transform_test = None
        self.set_transform_based_on_training_mode()

        self.path_text_file_for_this_task = None

        if self._train_mode:
            downloaded_list = self.train_list
            
        else:
            downloaded_list = self.test_list

        self.images = []
        self.labels_real = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.images.append(entry['data'])
                if 'labels' in entry:
                    self.labels_real.extend(entry['labels'])
                else:
                    self.labels_real.extend(entry['fine_labels'])

        self.images = np.vstack(self.images).reshape(-1, 3, 32, 32)
        self.images = self.images.transpose((0, 2, 3, 1))  # convert to HWC

        self.labels_real = np.asarray(self.labels_real)

        num_shots = self.configs.configs_dataset.num_shots if train and task_id > 0 else -1
        
        if train:
            file_name_prefix = 'train'
        else:
            file_name_prefix = 'test'

        file_name = file_name_prefix + f'_samples_for_task={task_id + 1}.txt'

        self.path_text_file_for_this_task = os.path.join(self.configs.directory_permutation_files, file_name)

        if not os.path.exists(self.path_text_file_for_this_task):
            self.create_session_file(file_name, num_shots)

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
        label_translation = [0] * len(self.configs.class_permutation)
        index = 0
        for i in self.configs.class_permutation.tolist():
            label_translation[i] = index
            index += 1

        self.label_translation_list = label_translation
    
    def create_session_file(self, file_name: str, num_shots=-1):
        self.configs.set_seed()
        selected_indices_all_list = []

        for label in self.selected_classes:
            selected_indices = np.where(label == self.labels_real)[0]
            if num_shots > 0:
                np.random.shuffle(selected_indices)
                selected_indices = selected_indices[:num_shots]
                assert len(selected_indices) == num_shots

            selected_indices_all_list += selected_indices.tolist()
        
        with open(self.path_text_file_for_this_task, "w") as text_file:
            for index in selected_indices_all_list:
                text_file.write(f'{index}\n')
        text_file.close()

        self.configs.logger.info(f"The session file \"{self.path_text_file_for_this_task}\" is created for the first time!")

    def use_samples_in_the_text_file(self):
        indices_selected = open(self.path_text_file_for_this_task).read().splitlines()
        indices_selected = [int(i) for i in indices_selected]
        self.labels_real = self.labels_real[indices_selected]

        # We verify the labels of the samples in the current file to belong to the classes for the current task.
        selected_labels_set = set(np.unique(self.labels_real).tolist())
        assert selected_labels_set == set(self.selected_classes)
        
        self.images = self.images[indices_selected]
        self.configs.logger.info(f"The session file \"{self.path_text_file_for_this_task}\" is loaded!")

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            msg = 'Dataset metadata file not found or corrupted. You can use download=True to download it'
            self.configs.logger.error(msg)
            raise RuntimeError(msg)
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        image = self.images[index]
        label_real = self.labels_real[index]
        labels_translated = self.label_translation_list[label_real]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        image = Image.fromarray(image)

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            label_real = self.target_transform(label_real)

        return image, labels_translated, label_real

    def __len__(self):
        return len(self.images)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self):
        return "Split: {}".format("Train" if self._train_mode is True else "Test")
    

if __name__ == "__main__":
    dataroot = '~/dataloader/data/'
    batch_size_base = 128
    txt_path = "../../data/index_list/cifar100/session_2.txt"
    # class_index = open(txt_path).read().splitlines()
    class_index = np.arange(60)
    trainset = CIFAR100(root=dataroot, train=True, download=True, transform=None, selected_classes=class_index, base_session=True)
    testset = CIFAR100(root=dataroot, train=False, download=False, selected_classes=class_index, base_session=True)
    cls = np.unique(trainset.labels_real)
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_base, shuffle=True, num_workers=8,
                                              pin_memory=False)
    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=100, shuffle=False, num_workers=8, pin_memory=False, drop_last=False)
    print(testloader.dataset.data.shape)
