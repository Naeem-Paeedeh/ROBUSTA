# We pretrain the model on ImageNet only for CUB-100 experiments.

import os
from torchvision import datasets, transforms
from torchvision.datasets import ImageNet
from configs.Configurations import Configurations


def get_dataset_train_for_imagenet(configs: Configurations, dino: bool):
    image_size = configs.image_size

    if dino:
        transform_train = configs.dino_transforms
    else:
        transform_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )

    root = os.path.expanduser(configs.configs_dataset.dataroot)
    
    return datasets.ImageFolder(root=os.path.join(root, 'train'), transform=transform_train)
    # We may have used the following to train the model in the first part
    # return ImageNet(root=root, split="train", transform=transform_train)


def get_dataset_test_for_imagenet(configs: Configurations):
    image_size = configs.image_size

    transform_test = transforms.Compose(
        [
            # This following line is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate
            transforms.Resize([int(image_size * 1.15), int(image_size * 1.15)]),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )

    root = os.path.expanduser(configs.configs_dataset.dataroot)

    return datasets.ImageFolder(root=os.path.join(root, 'validation'), transform=transform_test)
    # We may have used the following to train the model in the first part
    # return ImageNet(root=root, split="val", transform=transform_test)
