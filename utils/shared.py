import argparse
import torch
import torchvision
import time
import numpy as np
import torchvision.transforms
from torch.utils.data import Dataset
import os
import logging
from box import Box
import re
from torch.optim import lr_scheduler, SGD, Adam, AdamW, Rprop
from torch import Tensor as T


class DatasetFromTensors(Dataset):
    def __init__(self, inputs: torch.Tensor, labels: torch.Tensor, transform=None):
        super().__init__()
        self.data = inputs
        self.labels = labels
        self.n_samples = len(self.labels)
        self.index = 0
        self.transform = transform

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        sample = self.data[index]
        if self.transform is not None:
            sample = torchvision.transforms.ToPILImage()(sample)
            sample = self.transform(sample)
        return sample, self.labels[index]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class BestResult:
    def __init__(self) -> None:
        self.max = 0.0
        self.epoch = -1

    def update(self, value, epoch):
        if value >= self.max:   # Equal means that we prefer the latest model
            self.max = value
            self.epoch = epoch
            return True
        return False

    def to_string(self, times_100: bool = True):
        """_summary_

        Args:
            times_100 (bool, optional): _description_. Defaults to True.
            percent_sign (bool, optional): Is the value indicates percent or other number. Defaults to True.

        Returns:
            _type_: _description_
        """
        acc = self.max * 100.0 if times_100 else self.max
        return f"The best test accuracy was {acc:.2f}% at epoch {self.epoch}"
    

class BestModel:
    def __init__(self, max_or_min: str) -> None:
        self._best_value_for_criterion = 0.0
        self._best_state = None
        self._first_time = True
        self.max_or_min = max_or_min
        assert self.max_or_min in ['max', 'min']

    def update(self, model, criterion_value: float | int):
        # Equal the best means that we prefer the latest model
        if (self.max_or_min == 'max' and criterion_value >= self._best_value_for_criterion) or (self.max_or_min == 'min' and criterion_value <= self._best_value_for_criterion) or self._first_time:
            self._best_value_for_criterion = criterion_value
            self._best_state = model.state_dict()
            self._first_time = False
            return True     # Updated!
        return False        # Nothing is changed!
    
    def best_state(self):
        return self._best_state


class Stopwatch:
    """
    Stopwatch computes the time between start and stop.
    Then we can add time to the total_elapsed_time dictionary by watch name.
    """
    def __init__(self, keys: list = None):
        if keys is None:
            keys = []
        self._start_time = {k: time.time() for k in keys}

    def reset(self, key):
        self._start_time[key] = time.time()

    def elapsed_time(self, key):
        if key in self._start_time:
            return time.time() - self._start_time[key]

        self.reset(key)
        return 0.0

    @staticmethod
    def convert_to_hours_minutes(time_in_seconds: float) -> str:
        time_in_seconds = int(time_in_seconds)
        days = time_in_seconds // (24 * 3600)
        hours = (time_in_seconds % (24 * 3600)) // 3600
        minutes = (time_in_seconds % 3600) // 60
        seconds = time_in_seconds % 60

        def plural(x):
            if x != 1:
                return 's'
            return ''

        res_list = []

        if days > 0:
            res_list.append(f"{days} day{plural(days)}")
        if hours > 0:
            res_list.append(f"{hours} hour{plural(hours)}")
        if minutes > 0:
            res_list.append(f"{minutes} minute{plural(minutes)}")
        res_list.append(f"{seconds} second{plural(seconds)}")

        if len(res_list) == 1:
            return res_list[0]
        elif len(res_list) == 2:
            return ' and '.join(res_list)
        else:
            res = ', '.join(res_list[:-1]) + f', and {res_list[-1]}'
            
        return res

    def elapsed_time_in_hours_minutes(self, key):
        return self.convert_to_hours_minutes(self.elapsed_time(key))

    def __getitem__(self, name):
        return self.elapsed_time(name)

    def __getattr__(self, name: str):
        return self.elapsed_time(name)


class MovingAverageDict:
    def __init__(self, capacity, logger=None):
        self.capacity = capacity
        self.meters: dict[str, _MovingAverage] = {}
        self.logger = logger
        self.logging_method = print if logger is None else logger.info

    def __getitem__(self, key):
        if key in self.meters:
            return self.meters[key]
        msg = f"Error: You didn't define or add a value to the {key} key!"
        if self.logger is not None:
            self.logger.exception(msg)
        raise Exception(msg)

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))

            if k not in self.meters:
                self.meters[k] = _MovingAverage(self.capacity)
            self.meters[k].update(v)

    def reset_all(self):
        for meter in self.meters.values():
            meter.reset()

    def display(self, logger=None):
        for key, val in self.meters.items():
            output = f"Average {key} for the last {val.count} iterations: {val.calculate()}"
            self.logging_method(output)


class _MovingAverage:
    def __init__(self, capacity):
        self.capacity = capacity
        self.array = np.zeros(self.capacity)
        self.ind = 0
        self.count = 0
        self.sum = 0.0

    def update(self, x):
        self.sum += x - self.array[self.ind]
        self.array[self.ind] = x
        self.ind = (self.ind + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def calculate(self):
        return self.sum / self.count

    def reset(self):
        self.array = np.zeros(self.capacity)
        self.ind = 0
        self.count = 0
        self.sum = 0


class AveragerDict(object):
    def __init__(self, logger=None):
        self.meters = {}
        self.logging_method = print if logger is None else logger.info

    def __getitem__(self, key):
        if key not in self.meters:
            self.meters[key] = _Averager()
        return self.meters[key]

    def add(self, key: str, value: float | int | torch.Tensor, n: int = 1):
        if isinstance(value, torch.Tensor):
            value = value.item()
        assert isinstance(value, (float, int))

        if key not in self.meters:
            self.meters[key] = _Averager()
            
        self.meters[key].add(value, n)

    def reset_all(self):
        for meter in self.meters.values():
            meter.reset()

    def display(self):
        for key, val in self.meters.items():
            output = f"The average {key} for {val.count} items is {val.calculate()}"
            self.logging_method(output)


class _Averager():
    def __init__(self):
        self.sum = 0
        self.data = []

    def reset(self):
        self.sum = 0
        self.data = []

    def add(self, val: float | int, n: int = 1):
        self.sum += val * n
        for _ in range(n):
            self.data.append(val)

    def calculate(self):
        return self.sum / len(self.data)

    def __len__(self):
        return len(self.data)


class verification:
    """This class verifies the changes in a tensor's content.
    """
    def __init__(self, configs, old_value: T, must_be_changed: bool, error_message: str) -> None:
        self.configs = configs
        old_value = self.use_data_tensor_inside(old_value)
        self.old_value = old_value.detach().clone()
        self.must_change_or_not = must_be_changed
        self.error_message = error_message
        
    @staticmethod
    def use_data_tensor_inside(x):
        if hasattr(x, 'data'):
            return x.data
        return x

    def verify(self, current_value: T):
        current_value = self.use_data_tensor_inside(current_value)
        with torch.no_grad():
            change = (current_value.detach().clone() - self.old_value).abs().sum()
            if (self.must_change_or_not and change == 0.0) or (not self.must_change_or_not and change != 0.0):
                self.configs.logger.exception(self.error_message)
                raise Exception(self.error_message)
        # If there is no problem, we just pass!
    

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def parse_str_bool(x):
    if isinstance(x, bool):
        return x
    x = x.lower()
    if x in ('yes', 'y', 'true', 't', '1', 'on'):
        return True
    
    if x in ('no', 'n', 'false', 'f', '0', 'off'):
        return False
    
    msg = 'Boolean value is expected!'
    raise argparse.ArgumentTypeError(msg)


# From the https://github.com/cpphoo/STARTUP
class AverageMeterSet_STARTUP:
    def __init__(self):
        self.meters = {}

    def __getitem__(self, key):
        return self.meters[key]

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))

            if k not in self.meters:
                self.meters[k] = AverageMeter_STARTUP()
            self.meters[k].update(v)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self, postfix=''):
        return {name + postfix: meter.value for name, meter in self.meters.items()}

    def averages(self, postfix='/avg'):
        return {name + postfix: meter.average for name, meter in self.meters.items()}

    def sums(self, postfix='/sum'):
        return {name + postfix: meter.sum for name, meter in self.meters.items()}

    def counts(self, postfix='/count'):
        return {name + postfix: meter.count for name, meter in self.meters.items()}


# From the https://github.com/cpphoo/STARTUP
class AverageMeter_STARTUP:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.value = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.value = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        """
        value is the average value
        n : the number of items used to calculate the average
        """
        self.value = value
        self.sum += value * n
        self.count += n
        self.average = self.sum / self.count


# From the https://github.com/cpphoo/STARTUP
def accuracy_STARTUP(logits, ground_truth, top_k=None):
    if top_k is None:
        top_k = [1, ]
    assert len(logits) == len(ground_truth)
    # this function will calculate per class acc
    # average per class acc and acc

    n, d = logits.shape

    label_unique = torch.unique(ground_truth)
    acc = {'average': torch.zeros(len(top_k)),
           'per_class_average': torch.zeros(len(top_k)),
           'per_class': [[] for _ in label_unique],
           'gt_unique': label_unique,
           'topk': top_k,
           'num_classes': d}

    max_k = max(top_k)
    argsort = torch.argsort(logits, dim=1, descending=True)[:, :min([max_k, d])]
    correct = (argsort == ground_truth.view(-1, 1)).float()

    for indi, i in enumerate(label_unique):
        ind = torch.nonzero(ground_truth == i, as_tuple=False).view(-1)
        correct_target = correct[ind]

        # calculate topk
        for indj, j in enumerate(top_k):
            num_correct_partial = torch.sum(correct_target[:, :j]).item()
            acc_partial = num_correct_partial / len(correct_target)
            acc['average'][indj] += num_correct_partial
            acc['per_class_average'][indj] += acc_partial
            acc['per_class'][indi].append(acc_partial * 100)

    acc['average'] = acc['average'] / n * 100
    acc['per_class_average'] = acc['per_class_average'] / len(label_unique) * 100

    return acc


# We mostly follow the name conventions of the STARTUP and TVT for argument names.
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings_file', type=str, default='', required=True,
                        help='The path to the TOML file for settings.')

    args = parser.parse_args()
    return args


# Inspired by the F2M implementation.
# {
def get_root_logger(logger_name='FSCIL',
                    log_level=logging.INFO,
                    log_file=None):
   
    logger = logging.getLogger(logger_name)
    # if the logger has been initialized, just return it
    if logger.hasHandlers():
        return logger

    format_str = '%(asctime)s:%(levelname)s:%(name)s:%(lineno)d: %(message)s'
    logging.basicConfig(format=format_str, level=log_level)
    rank, _ = 0, 1
    if rank != 0:
        logger.setLevel('ERROR')
    elif log_file is not None:
        file_handler = logging.FileHandler(log_file, 'w')
        file_handler.setFormatter(logging.Formatter(format_str))
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)

    return logger


def get_env_info():
    """Get environment information.

    Currently, only log the software version.
    """
    msg = ('\nVersion Information: \n\tPyTorch: %s\n\tTorchVision: %s', torch.__version__, torchvision.__version__)
    return msg


def get_time_str(add_time: bool = True):
    my_str = 'Date_%Y-%m-%d'
    if add_time:
        my_str += ',Time_%H-%M-%S'
    return time.strftime(my_str, time.localtime())


def Box2str(dm_object: Box, indent_level=1) -> str:
    """dict to string for printing options.

    Args:
        dm_object (Box): Box object.
        indent_level (int): Indent level. Default: 1.

    Return:
        (str): Option string for printing.
    """
    msg = ' {\n'

    for k, v in dm_object.items():
        if k.startswith('_'):
            continue
        if isinstance(v, dict):
            msg += ' ' * (indent_level * 2) + k + ' ='
            msg += Box2str(v, indent_level + 1)
            msg += ' ' * (indent_level * 2) + '\n'
        else:
            if isinstance(v, str):
                msg += ' ' * (indent_level * 2) + k + ' = \"' + v + '\"\n'
            else:
                msg += ' ' * (indent_level * 2) + k + ' = ' + str(v) + '\n'

    msg += ' ' * ((indent_level - 1) * 2) + '}'

    return msg
# }


def convert_none_string_to_None(bx: Box) -> Box:
    """It converts the "none" string values in the TOML files to real None values of Python.

    Args:
        dm (Box): Our box object from the TOML file.

    Returns:
        _type_: _description_
    """
    for k, v in bx.items():
        if isinstance(bx[k], Box):
            bx[k] = convert_none_string_to_None(v)
        elif bx[k] == "none":
            bx[k] = None
    return bx


def print_estimated_remaining_time(total_time, total_tasks, num_finished_tasks, display_func=print):
    num_finished_tasks_from_one = num_finished_tasks + 1
    ert = (total_tasks - num_finished_tasks_from_one) * total_time / num_finished_tasks_from_one
    display_func("Estimated remaining time: %s" % Stopwatch.convert_to_hours_minutes(ert))
    

def _find_a_pattern(string: str, pattern: str):
    if string is None or string == "":
        return string
    match = re.search(pattern, string)
    if match:
        substring = match.group()
        return substring
    return None


def find_patterns_sequencially(string, pattern_list: list):
    res = string
    for pt in pattern_list:
        res = _find_a_pattern(res, pt)
    return res


def prepare_scheduler(scheduler_configs: Box, optimizer):
    if scheduler_configs.name == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer,
                                             milestones=scheduler_configs.milestones,
                                             gamma=scheduler_configs.gamma)
    elif scheduler_configs.name == "ReduceLROnPlateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   mode=scheduler_configs.mode,
                                                   factor=scheduler_configs.factor,
                                                   patience=scheduler_configs.patience,
                                                   cooldown=scheduler_configs.cooldown,
                                                   min_lr=scheduler_configs.min_lr,
                                                   verbose=scheduler_configs.verbose
                                                   )
    else:
        raise NotImplementedError
    
    return scheduler


def prepare_optimizer(configs_optimizer, parameters, task_id: int, logger) -> SGD | Adam | AdamW | Rprop:
    optimizer_name = configs_optimizer.optimizer_name
    is_incremental = task_id != -1
    
    if is_incremental:      # In the incremental learning tasks.
        lr = configs_optimizer.lr[task_id]
        weight_decay = configs_optimizer.weight_decay[task_id]
    else:                   # In the supervised learning
        lr = configs_optimizer.lr
        if optimizer_name in ['SGD', 'Adam', 'AdamW']:
            weight_decay = configs_optimizer.weight_decay

    if optimizer_name == "SGD":
        dampening = 0.0
        if 'dampening' in configs_optimizer:
            dampening = configs_optimizer.dampening
        optimizer = SGD(params=parameters,
                        lr=lr,
                        weight_decay=weight_decay,
                        momentum=configs_optimizer.momentum,
                        dampening=dampening)
    elif optimizer_name == "Adam":
        optimizer = Adam(params=parameters,
                         lr=lr,
                         weight_decay=weight_decay,
                         betas=(configs_optimizer.momentum,
                                configs_optimizer.momentum2))
    elif optimizer_name == "AdamW":
        optimizer = AdamW(params=parameters,
                          lr=lr,
                          weight_decay=weight_decay)
    elif optimizer_name == "Rprop":
        optimizer = Rprop(params=parameters,
                          lr=lr)
    else:
        logger.exception("Not implemented!")
        raise NotImplementedError

    return optimizer


def calculate_mean(x: T) -> T:  # Without worrying about overflowing
    return (x / x.numel()).sum()


def calculate_std(x: T) -> T:  # Without worrying about overflowing
    mean = calculate_mean(x)
    diff = x - mean
    sm_squared = diff * diff
    var = (sm_squared).sum() / (x.numel() - 1)
    return torch.sqrt(var)


def save_a_list_to_a_text_file(my_list: list, file_name: str):
    assert isinstance(my_list, list)
    str_list = [str(i) for i in my_list]
    temp_str = ','.join(str_list)
    with open(file_name, "w") as f:
        f.write(temp_str)
