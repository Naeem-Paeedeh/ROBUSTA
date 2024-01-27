import torch
from configs.Configurations import Configurations
from dataloader.mini_imagenet import MiniImageNet
from dataloader.cifar import CIFAR100
from dataloader.cub200 import CUB200
from dataloader.imagenet import get_dataset_test_for_imagenet, get_dataset_train_for_imagenet


# We modified the S3C implementation
def get_datasets_and_dataloaders(configs: Configurations, dino: bool, task_id=None):
    """It creates datasets and dataloaders for the current task.

    Args:
        configs (Configurations): Configuration
        dino (bool): To use the DINO transormations
        task_id (_type_, optional): Task-ID. Defaults to None.

    Returns:
        _type_: _description_
    """
    dataset_train, loader_train = get_dataloader_train(configs, dino, task_id)

    dataset_test, loader_test = get_dataloader_test(configs, dino, task_id)

    return dataset_train, loader_train, dataset_test, loader_test


def get_dataloader_train(configs: Configurations, dino: bool, task_id=None):
    if task_id is None:
        task_id = configs.get_task_id()
    selected_classes_train = configs.get_selected_classes("train", task_id)
    dataset_name = configs.configs_dataset.dataset_name

    if dataset_name == 'cifar100':
        dataset_train = CIFAR100(configs=configs, train=True, dino=dino, download=(task_id == 0), selected_classes=selected_classes_train, task_id=task_id)
    elif dataset_name == 'cub200':
        dataset_train = CUB200(configs=configs, train=True, dino=dino, selected_classes=selected_classes_train, task_id=task_id)
    elif dataset_name == 'mini_imagenet':
        dataset_train = MiniImageNet(configs=configs, train=True, dino=dino, selected_classes=selected_classes_train, task_id=task_id)
    elif dataset_name == 'imagenet':
        dataset_train = get_dataset_train_for_imagenet(configs, dino=dino)

    if task_id == 0:
        batch_size_train = configs.batch_size_base
        drop_last_base = configs.configs_dataset.drop_last_base
    else:
        if configs.batch_size_new == 0:
            configs.batch_size_new = dataset_train.__len__()
        batch_size_train = configs.batch_size_new
        drop_last_base = False

    assert batch_size_train > 0

    loader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=batch_size_train, shuffle=True, num_workers=configs.configs_dataset.num_workers, pin_memory=False, drop_last=drop_last_base)
    
    return dataset_train, loader_train


def get_dataloader_test(configs: Configurations, dino: bool, task_id=None):
    if task_id is None:
        task_id = configs.get_task_id()
    selected_classes_test = configs.get_selected_classes("test", task_id)
    dataset_name = configs.configs_dataset.dataset_name

    if dataset_name == 'cifar100':
        dataset_test = CIFAR100(configs=configs, train=False, dino=dino, download=False, selected_classes=selected_classes_test, task_id=task_id)
    elif dataset_name == 'cub200':
        dataset_test = CUB200(configs=configs, train=False, dino=dino, selected_classes=selected_classes_test, task_id=task_id)
    elif dataset_name == 'mini_imagenet':
        dataset_test = MiniImageNet(configs=configs, train=False, dino=dino, selected_classes=selected_classes_test, task_id=task_id)
    elif dataset_name == 'imagenet':
        dataset_test = get_dataset_test_for_imagenet(configs)

    loader_test = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=configs.batch_size_test, shuffle=True, num_workers=configs.configs_dataset.num_workers, pin_memory=False, drop_last=False)
    
    return dataset_test, loader_test

