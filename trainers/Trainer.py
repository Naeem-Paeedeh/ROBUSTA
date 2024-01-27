# Please refer to the end of the file for references.

from __future__ import print_function
from configs.Configurations import Configurations


class Trainer:
    def __init__(self, configs: Configurations):
        self.configs = configs
        self.device = self.configs.device

    def learn_this_task(self, loader_train, loader_test):    # def new_task(
        raise NotImplementedError

    def training(self, loader_train, loader_test=None):
        raise NotImplementedError
    
    # It was def evaluating(
    def evaluation(self, loader_test):
        raise NotImplementedError


# @inproceedings{wang2023rehearsal,
#   title={Rehearsal-free Continual Language Learning via Efficient Parameter Isolation},
#   author={Wang, Zhicheng and Liu, Yufang and Ji, Tao and Wang, Xiaoling and Wu, Yuanbin and Jiang, Congcong and Chao, Ye and Han, Zhencong and Wang, Ling and Shao, Xu and others},
#   booktitle={Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
#   pages={10933--10946},
#   year={2023}
# } -> https://github.com/Dicer-Zz/EPI
