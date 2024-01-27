# We utilized the EPI
# @inproceedings{wang2023rehearsal,
#   title={Rehearsal-free Continual Language Learning via Efficient Parameter Isolation},
#   author={Wang, Zhicheng and Liu, Yufang and Ji, Tao and Wang, Xiaoling and Wu, Yuanbin and Jiang, Congcong and Chao, Ye and Han, Zhencong and Wang, Ling and Shao, Xu and others},
#   booktitle={Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
#   pages={10933--10946},
#   year={2023}
# }

from __future__ import print_function
from copy import deepcopy
from enum import Enum
import torch
from torch import nn
from models.ViT_CCT import StochasticClassifier, ViT_CCT, BN_3D, BN_Pixels
from utils import dino_utils
from configs.Configurations import Configurations
import torch.nn.functional as F
from torch import Tensor as T
from models.PredictionNet import PredictionNet


def mahalanobis(queries, mean, cov_inv, norm=2):
    """
    args:
        querys: [n, dim]
        mean: [dim]
        cov_inv: [dim, dim]
    return:
        [n]
    """
    diff = queries - mean
    maha_dis = (diff @ cov_inv) * diff

    if norm == 2:
        return maha_dis.sum(dim=1)
    if norm == 1:
        return maha_dis.abs().sqrt().sum(dim=1)
    if norm == 'inf':
        return maha_dis.max(dim=1)
    

class task_id_detection_mode(Enum):
    predict_task_id = 1
    oracle = 2
    current_task_id = 3
    previous_task_id = 4
    no_prefix_or_prompt = 5


class Statistics(nn.Module):
    def __init__(self, configs: Configurations, embed_dim: int, use_covariance: bool, use_shared_covariance: bool) -> None:
        super().__init__()
        self.configs = configs
        self.device = configs.device
        self.embed_dim = embed_dim
        self.count = 0
        self.store_covariance: bool = use_covariance
        self.last_added_covariance = None       # We must subtract it when we update the stats.

        self.means_ParameterList = nn.ParameterList()
        self.labels_space_or_task_id_ParameterList = nn.ParameterList()

        if self.store_covariance:
            self.use_shared_covariance: bool = use_shared_covariance
            
            if self.use_shared_covariance:
                self.accumulated_shared_covariances = nn.Parameter(torch.zeros(embed_dim, embed_dim, device=configs.device), requires_grad=False)

                self.shared_covariances_inverse = nn.Parameter(torch.ones(embed_dim, embed_dim), requires_grad=False)
            else:
                self.covariances_for_each_task = nn.ParameterList()
                self.covariance_inverses_for_each_task = nn.ParameterList()

    def store_and_accumulate_the_statistics_for_this_task(self, means: T, covariance: T, labels_space_or_task_id: T):
        self.count += 1
        self.means_ParameterList.append(nn.Parameter(means.to(self.device), requires_grad=False))
        self.labels_space_or_task_id_ParameterList.append(nn.Parameter(labels_space_or_task_id.to(self.device), requires_grad=False))
        
        if covariance is not None:      # For Mahalanobis distance
            if self.use_shared_covariance:
                # To store and accumulates the statistics of the current object.
                self.accumulated_shared_covariances += covariance
                self.last_added_covariance = covariance.detach().clone()
                cov_inv_temp = torch.linalg.pinv(self.accumulated_shared_covariances / self.count, hermitian=True)
                self.shared_covariances_inverse = nn.Parameter(cov_inv_temp.to(self.device), requires_grad=False)
            else:
                self.covariances_for_each_task.append(nn.Parameter(covariance.to(self.device), requires_grad=False))
                cov_inv_temp = torch.linalg.pinv(covariance, hermitian=True)
                self.covariance_inverses_for_each_task.append(nn.Parameter(cov_inv_temp.to(self.device), requires_grad=False))

    def update_the_statistics_for_the_last_task(self, means, covariance, labels_space_or_task_id):
        self.means_ParameterList[-1] = nn.Parameter(means.to(self.device), requires_grad=False)
        self.labels_space_or_task_id_ParameterList[-1] = nn.Parameter(labels_space_or_task_id.to(self.device), requires_grad=False)
        
        if covariance is not None:      # For Mahalanobis distance
            if self.use_shared_covariance:
                self.accumulated_shared_covariances += covariance - self.last_added_covariance
                self.last_added_covariance = covariance.detach().clone()
                cov_inv_temp = torch.linalg.pinv(self.accumulated_shared_covariances / self.count, hermitian=True)
                self.shared_covariances_inverse = nn.Parameter(cov_inv_temp.to(self.device), requires_grad=False)
            else:
                self.covariances_for_each_task[-1] = nn.Parameter(covariance.to(self.device), requires_grad=False)
                cov_inv_temp = torch.linalg.pinv(covariance, hermitian=True)
                self.covariance_inverses_for_each_task[-1] = nn.Parameter(cov_inv_temp.to(self.device), requires_grad=False)

    def get_means_and_covariance_inverse_and_labels_space_or_task_id(self, task_id: int):
        means = self.means_ParameterList[task_id]
        labels_space_or_task_id = self.labels_space_or_task_id_ParameterList[task_id]

        if self.store_covariance:
            if self.use_shared_covariance:
                cov_inv = self.shared_covariances_inverse
            else:
                cov_inv = self.covariance_inverses_for_each_task[task_id]
        else:
            cov_inv = None
        
        return means, cov_inv, labels_space_or_task_id
    
    def get_means(self, task_id: int):
        means = self.means_ParameterList[task_id]
        
        return means
    

def freeze_it(model) -> None:
    if model is None:
        return
    for param in model.parameters():
        param.requires_grad = False
    for m in model.modules():
        if type(m) in [BN_3D, BN_Pixels, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]:
            m.requires_grad = False
            m.eval()


class EPI(nn.Module):
    def __init__(self, configs: Configurations):
        super().__init__()

        self.configs = configs
        self.backbone: ViT_CCT = configs.get_the_model()
        self.classifier_head: StochasticClassifier = configs.get_the_classifier_head()
        self.device = self.configs.device
        self.dropout = nn.Dropout(configs.configs_model.attention_dropout_rate)
        self.freeze_backbone = self.configs.configs_FSCIL.freeze_backbone
        
        if self.freeze_backbone:
            freeze_it(self.backbone)
        
        self.n_heads = nn.Parameter(torch.tensor(configs.configs_model.transformer_num_heads), requires_grad=False)
        self.embed_dim = nn.Parameter(torch.tensor(configs.configs_model.embed_dim), requires_grad=False)
        self.head_dim = nn.Parameter(torch.tensor(configs.configs_model.head_dim), requires_grad=False)
        self.prefix_or_prompt = self.configs.configs_FSCIL.configs_PEFT.prefix_or_prompt

        if self.prefix_or_prompt not in ['prefix', 'prompt', None]:
            msg = "Error: prefix_or_prompt's value is incorrect!"
            self.configs.logger.exception(msg)
            raise Exception(msg)

        self.prefixes = None
        self.prompts = None
        self.prefixes_count = None
        self.statistics_for_task_identification = None
        self.prediction_net_list = None
        self.configs_prediction_net = None
        self.prefix_seq_length = None
        self.separate_PredictionNet_for_each_task = False
        self.use_delta_parameters_for_base_task = False
        
        if self.prefix_or_prompt is not None:
            self.use_delta_parameters_for_base_task = self.configs.configs_FSCIL.use_delta_parameters_for_base_task
            self.prefixes = nn.ParameterList()
            self.prompts = nn.ParameterList()
            self.prefixes_count = nn.Parameter(torch.tensor(0.0), requires_grad=False)
            # When the PredictionNet is not ready, we use these statistics.
            self.enable_Mahalanobis_distance = self.configs.configs_FSCIL.enable_Mahalanobis_distance
            self.statistics_for_task_identification = Statistics(configs=configs,
                                                                embed_dim=self.embed_dim,
                                                                use_covariance=self.enable_Mahalanobis_distance,
                                                                use_shared_covariance=self.configs.configs_FSCIL.use_shared_covariance)
            self.prediction_net_list: list[PredictionNet] = configs.prediction_net_list
            self.configs_prediction_net = self.configs.configs_FSCIL.PredictionNet
            self.prefix_seq_length = nn.Parameter(torch.tensor(configs.configs_FSCIL.configs_PEFT.prefix_seq_length), requires_grad=False)
            self.separate_PredictionNet_for_each_task = self.configs_prediction_net.enabled and self.configs_prediction_net.separate_PredictionNet_for_each_task
            
        self.cached_one_hot_labels_for_MSE_loss = torch.eye(self.configs.configs_dataset.num_ways, device=self.device) * 2.0 - 1.0
    
    def get_prefix_shape(self, task_id: int):
        number_of_layers_for_prefixes = self.configs.configs_FSCIL.configs_PEFT.number_of_layers_for_prefixes[task_id]
        
        if number_of_layers_for_prefixes == -1:
            number_of_layers_for_prefixes = self.configs.configs_model.num_layers
        return (number_of_layers_for_prefixes, 2, self.n_heads, self.prefix_seq_length[task_id], self.head_dim)
    
    def get_prompts_shape(self, task_id: int):
        return (self.prefix_seq_length[task_id], self.embed_dim)
    
    def preparation_for_the_new_task(self):
        """Expands the prefix
        """
        # Freezing the previous prefixes or prompts
        task_id = self.configs.get_task_id()
        freeze_it(self.prefixes)

        # We do not want to use any prefix for the base task. We just consider its statistics
        if self.prefix_or_prompt is not None:
            if self.use_delta_parameters_for_base_task or self.configs.get_task_id() > 0:
                prefixes_or_prompts_tensor = self.initialize_prefixes_or_prompts_tensor()  # Its shape is (pre_seq_len, num_hidden_layers * hidden_size * 2) for prefix model
                if self.prefix_or_prompt == 'prefix':
                    self.prefixes.append(nn.Parameter(prefixes_or_prompts_tensor, requires_grad=True))
                elif self.prefix_or_prompt == 'prompt':
                    self.prompts.append(nn.Parameter(prefixes_or_prompts_tensor, requires_grad=True))

            if self.configs_prediction_net.enabled:
                if self.separate_PredictionNet_for_each_task:
                    if self.configs_prediction_net.use_PredictionNet_for_this_task[task_id]:
                        prediction_net_for_this_task = self.initialize_a_new_prediction_net()
                        
                        if len(self.prediction_net_list) >= 1 and self.configs_prediction_net.remember_from_previous_task and self.prediction_net_list[-1] is not None:
                            self.configs.logger.info("The PredictionNet is copied from the previous task.")
                            state_prediction_net_previous_task = deepcopy(self.prediction_net_list[-1].state_dict())
                            prediction_net_for_this_task.load_state_dict(state_prediction_net_previous_task)
                                
                        self.prediction_net_list.append(prediction_net_for_this_task)
                    else:
                        self.prediction_net_list.append(None)
                else:       # If we have a single shared PredictionNet
                    if len(self.prediction_net_list) == 0:
                        self.prediction_net_list.append(self.initialize_a_new_prediction_net())

    def initialize_a_new_prediction_net(self):
        new_prediction_net = PredictionNet(self.configs.configs_model.embed_dim,
                                          n_layers=self.configs_prediction_net.n_layers,
                                          size_hidden_layer=self.configs_prediction_net.size_hidden_layer,
                                          bias=self.configs_prediction_net.bias,
                                          dropout_rate=self.configs_prediction_net.dropout_rate,
                                          use_real_residual_connections=self.configs_prediction_net.use_real_residual_connections,
                                          use_stochastic_classifier=False,
                                          device=self.device)
        return new_prediction_net
    
    def initialize_prefixes_or_prompts_tensor(self) -> T:
        task_id = self.configs.get_task_id()
        
        assert self.use_delta_parameters_for_base_task or task_id > 0
        
        fusion_mode = self.configs.configs_FSCIL.configs_PEFT.fusion_mode

        if fusion_mode == 'mean':
            raise NotImplementedError

        # random_init == True means that we initalize the current prefix with randn
        random_init = False
        
        if fusion_mode == 'random':
            random_init = True
        
        if fusion_mode == 'last':
            random_init = task_id == 0 or (task_id == 1 and not self.use_delta_parameters_for_base_task) or (task_id > 0 and self.get_prefix_shape(task_id) != self.get_prefix_shape(task_id - 1))
        
        # We freeze the previous prefixes and prompts
        freeze_it(self.prefixes)
        freeze_it(self.prompts)

        if self.prefix_or_prompt == 'prefix':
            if random_init:
                self.configs.logger.info(f"Prefixes are randomly initialized for task {task_id}.")
                new_prefixes_tensor = torch.randn(*self.get_prefix_shape(task_id=task_id), device=self.device)
            elif fusion_mode == 'last':      # 'last' means that we copy the prefixes from the previous task.
                self.configs.logger.info(f"Prefixes are copied from task {task_id - 1}.")
                new_prefixes_tensor = self.prefixes[-1].data.detach().clone()
            elif fusion_mode == 'zeros':
                self.configs.logger.info(f"Prefixes are initialized with zeros for task {task_id}.")
                new_prefixes_tensor = torch.zeros(*self.get_prefix_shape(task_id=task_id), device=self.device)

            return new_prefixes_tensor  # Its shape is (pre_seq_len, num_layers * embed_dim * 2) for prefix model
        elif self.prefix_or_prompt == 'prompt':
            if random_init:
                new_prompts_data = torch.randn(*self.get_prompts_shape(task_id=task_id), device=self.device)
            elif fusion_mode == 'last':
                self.configs.logger.info(f"Prompt is copied from task {task_id - 1}.")
                new_prompts_data = self.prompts[-1].data.detach().clone()
            elif fusion_mode == 'zeros':
                self.configs.logger.info(f"Prompt is initialized with zeros for task {task_id}.")
                new_prefixes_tensor = torch.zeros(*self.get_prompts_shape(task_id=task_id), device=self.device)
                
            return new_prompts_data
        
    def _get_embeddings(self,
                        samples: T,
                        task_ids_for_prefixes: T,
                        use_prediction_net: bool,
                        use_prefixes: bool = True
                        ) -> T:
        """This method obtain the embeddings for each sample by considering its prefixes except the base task's samples.

        Args:
            samples (T): _description_
            indices_for_prefixes_or_prompts (list): Indices of prefixes we must apply for each samples

        Returns:
            T: _description_
        """
        def index_select(parameter_list_object, task_ids: T):
            """If all samples have the same sequence length we can stack the prefixes together

            Args:
                parameter_list_object (_type_): _description_
                task_ids (T): _description_

            Returns:
                _type_: _description_
            """
            seq_length_set = set([self.prefix_seq_length[task_id].item() for task_id in task_ids])
            if len(seq_length_set) != 1:
                msg = "Error: We cannot stack the prefixes with different length together!"
                self.configs.logger.exception(msg)
                raise Exception(msg)
            
            chosen_items_list = []
            for ind in task_ids.tolist():
                chosen_items_list.append(parameter_list_object[ind])
            return torch.stack(chosen_items_list)

        if task_ids_for_prefixes is None or not use_prefixes or self.prefix_or_prompt is None:
            embeddings_final = self.backbone(samples)
        else:   # We separate the samples from all tasks. If the base task does not have prefixes we forward the base samples without prefixes. Moreover, We forward the incremenatl samples separately if the their sequence_lengths are different.
            indices_each_task = [None] * self.configs.configs_dataset.num_tasks
            indices_of_prefixes_for_each_task = [None] * self.configs.configs_dataset.num_tasks
            
            for task_id in task_ids_for_prefixes.unique():
                indices_each_task[task_id] = (task_ids_for_prefixes == task_id).nonzero(as_tuple=True)[0]
            
            for task_id in task_ids_for_prefixes.unique():
                if self.use_delta_parameters_for_base_task:
                    # Prefix index = the task_id
                    indices_of_prefixes_for_each_task[task_id] = task_ids_for_prefixes[indices_each_task[task_id]]
                elif task_id > 0:
                    # When we do not utilize any prefix for the base task, the prefix index = task_id - 1
                    indices_of_prefixes_for_each_task[task_id] = task_ids_for_prefixes[indices_each_task[task_id]] - 1
                    
            if self.prefix_or_prompt == 'prefix':
                embeddings_each_task = [None] * self.configs.configs_dataset.num_tasks
                for task_id in task_ids_for_prefixes.unique():
                    if len(indices_each_task[task_id]) > 0:
                        if task_id == 0 and not self.use_delta_parameters_for_base_task:
                            samples_base_task = samples[indices_each_task[0]]
                            embeddings_each_task[0] = self.backbone(samples_base_task)
                        else:
                            prefixes_for_this_task = index_select(self.prefixes, indices_of_prefixes_for_each_task[task_id])
                            # prefixes_this_tasks = prepare_prefixes(prefixes_for_this_task, self.dropout, )
                            prefixes_for_this_task = self.dropout(prefixes_for_this_task)
                            prefixes_this_tasks = prefixes_for_this_task.permute([1, 2, 0, 3, 4, 5])
                            samples_this_task = samples[indices_each_task[task_id]]
                            # 2 x (2, n_layers, batch_size, n_heads, prefix_sequence_length, head_dim)
                            # It should become (num_layer, 2, batch_szie, num_heads, prefix_sequence_length, head_dim)
                            embeddings_each_task[task_id] = self.backbone(samples_this_task, prefixes=prefixes_this_tasks)
            elif self.prefix_or_prompt == 'prompt':
                raise NotImplementedError
            
            embeddings_final = torch.zeros(samples.shape[0], self.embed_dim, device=self.device)
            for task_id in task_ids_for_prefixes.unique():
                embeddings_final[indices_each_task[task_id]] = embeddings_each_task[task_id]

        if self.prefix_or_prompt is not None and use_prediction_net and self.configs_prediction_net.enabled:
            with torch.no_grad():
                if self.separate_PredictionNet_for_each_task and self.configs_prediction_net.use_PredictionNet_for_this_task[task_id]:
                    assert task_ids_for_prefixes is not None     # Not implemented!
                    
                    for task_id in task_ids_for_prefixes.unique():
                        indices_for_tid = (task_ids_for_prefixes == task_id).nonzero(as_tuple=True)[0]
                        if self.prediction_net_list[task_id] is not None:
                            embeddings_final[indices_for_tid] = self.prediction_net_list[task_id](embeddings_final[indices_for_tid])
                else:
                    assert len(self.prediction_net_list) == 1
                    embeddings_final = self.prediction_net_list[0](embeddings_final)
        return embeddings_final

    def _predict_task_ids(self,
                          samples: T,
                          stats: Statistics,
                          use_prediction_net: bool) -> T:    # Finished!
        """
        arguments:
            embeddings: [bs, embed_dim]
        return:
            indices: [bs], indices of selected prompts
        """
        batch_size = samples.shape[0]

        scores_over_tasks = []
        
        with torch.no_grad():
            if self.enable_Mahalanobis_distance:
                if self.configs.configs_FSCIL.tasks_or_classes_for_Mahalanobis_distance_calculations == 'classes':
                    for task_id in range(stats.count):
                        means_over_classes, covariance_inverses, labels_space = stats.get_means_and_covariance_inverse_and_labels_space_or_task_id(task_id)

                        embeddings = self._get_embeddings(samples, torch.tensor([task_id] * batch_size), use_prediction_net, use_prefixes=self.configs.configs_FSCIL.use_prefixes_for_distance_calculations)
                        
                        num_labels, _ = means_over_classes.shape
                        
                        score_over_classes_list = []
                        for c in range(num_labels):
                            score = mahalanobis(embeddings, means_over_classes[c], covariance_inverses, norm=2)
                            score_over_classes_list.append(score)
                        # [num_labels, n]
                        score_over_classes = torch.stack(score_over_classes_list)
                        score, _ = score_over_classes.min(dim=0)

                        scores_over_tasks.append(score)
                    # [task_num, n]
                    scores_over_tasks = torch.stack(scores_over_tasks, dim=0)
                else:
                    scores_over_tasks = []
                    
                    for task_id in range(stats.count):
                        means, covariance_inverses, labels_space = stats.get_means_and_covariance_inverse_and_labels_space_or_task_id(task_id)

                        embeddings = self._get_embeddings(samples, torch.tensor([task_id] * batch_size), use_prediction_net)
                        
                        score = mahalanobis(embeddings, means, covariance_inverses, norm=2)
                        # [num_labels, n]
                        scores_over_tasks.append(score)
                    # [task_num, n]
                    scores_over_tasks = torch.stack(scores_over_tasks, dim=0)
                
                _, indices = torch.min(scores_over_tasks, dim=0)
            else:       # Euclidean distance
                assert self.configs.configs_dataset.num_shots == 1
                prototypes_tensor_list = []
                
                for task_id in range(stats.count):
                    means_over_classes = stats.get_means(task_id)
                    prototypes_tensor_list.append(means_over_classes)
                
                prototypes = torch.cat(prototypes_tensor_list, dim=0)

                embeddings = self._get_embeddings(samples, torch.tensor([task_id] * batch_size), use_prediction_net)
                
                distances = torch.cdist(embeddings, prototypes, p=2.0)
                _, labels = torch.min(distances, dim=1)
                indices = torch.tensor([self.configs.label_to_task_id_dictionary[i.item()] for i in labels], device=self.device)
            return indices
    
    def infer_task_ids_from_labels(self, labels):
        task_ids = torch.zeros_like(labels)
        for i in range(1, self.configs.get_task_id() + 1):
            labels_start = self.configs.get_this_task_range(i)[0]
            task_ids += labels >= labels_start

        return task_ids

    def forward(self,
                samples,
                task_id_mode: task_id_detection_mode,
                use_prediction_net_for_task_id: bool,
                use_prediction_net_for_embeddings: bool,
                labels=None,        # Oracle mode requires the labels
                ) -> tuple[T, T]:
        
        batch_size = samples.shape[0]

        if self.training and task_id_mode != task_id_detection_mode.current_task_id:
            msg = 'Error: Invalid arguments!'
            self.configs.logger.exception(msg)
            raise Exception(msg)
        
        if task_id_mode == task_id_detection_mode.current_task_id:  # or "if self.training:""
            task_ids = torch.tensor([self.configs.get_task_id()] * batch_size)
        elif task_id_mode == task_id_detection_mode.oracle:                 # If our model knows the actual task-ids
            assert labels is not None
            task_ids = self.infer_task_ids_from_labels(labels)
        elif task_id_mode == task_id_detection_mode.predict_task_id:        # To detect the task ID for evaluation
            task_ids = None
            if self.prefix_or_prompt in ['prefix', 'prompt']:               # We do not need task-ids when prefixes are disabled
                task_ids = self._predict_task_ids(samples,
                                                stats=self.statistics_for_task_identification,
                                                use_prediction_net=use_prediction_net_for_task_id)
        elif task_id_mode == task_id_detection_mode.current_task_id:
            task_ids = torch.tensor([self.configs.get_task_id()] * batch_size)
        elif task_id_mode == task_id_detection_mode.previous_task_id:
            assert self.configs.get_task_id() > 0
            task_ids = torch.tensor([self.configs.get_task_id() - 1] * batch_size)
        elif task_id_mode == task_id_detection_mode.no_prefix_or_prompt:      # To ignore the prefixes or prompts
            task_ids = None

        embeddings = self._get_embeddings(samples, task_ids, use_prediction_net_for_embeddings)

        return embeddings, task_ids

    def calculate_logits(self, embeddings, task_ids, ignore_logits_for_other_tasks=True):
        if isinstance(self.classifier_head, StochasticClassifier) and not self.training:
            logits = self.classifier_head(embeddings, self.configs.configs_FSCIL.evaluation.stochastic)
        else:
            logits = self.classifier_head(embeddings)

        if ignore_logits_for_other_tasks and task_ids is not None:
            # To ensure that other classes from other tasks will not be selected for the detected task.
            min_value = -1e4
            
            for i, task_id in enumerate(task_ids):
                start, end = self.configs.get_this_task_range(task_id)
                logits[i][:start] = min_value
                logits[i][end:] = min_value

        return logits

    def calculate_loss(self, logits, labels):
        loss = None
        
        if labels is not None:
            rng = self.configs.get_this_task_range()

            logits = logits[:, rng[0]:rng[1]]
            labels_from_zero_for_current_task = labels.detach().clone()
            labels_from_zero_for_current_task -= rng[0]

            loss = F.cross_entropy(logits, labels_from_zero_for_current_task)
            
        return loss
    
    def train(self, mode=True):
        super().train(mode)
        task_id = self.configs.get_task_id()
        
        self.backbone.train(mode=not self.freeze_backbone)
        self.classifier_head.train(mode=self.configs.configs_FSCIL.optimizer.lr_head[task_id] > 0)
        self.dropout.train()
        
        if self.prefix_or_prompt == 'prefix':
            for id in range(len(self.prefixes)):
                self.prefixes[id].requires_grad = id == len(self.prefixes) - 1
        elif self.prefix_or_prompt == 'prompt':
            for id in range(len(self.prompts)):
                self.prompts[id].requires_grad = id == len(self.prompts) - 1
        elif self.prefix_or_prompt is not None:
            msg = "Incorrect prefix_or_prompt mode!"
            self.configs.logger.exception(msg)
            raise Exception(msg)

    def eval(self):
        super().eval()
        self.backbone.eval()
        self.classifier_head.eval()
        self.dropout.eval()
        
        if self.prefix_or_prompt == 'prefix':
            for id in range(len(self.prefixes)):
                self.prefixes[id].requires_grad = False
        elif self.prefix_or_prompt == 'prompt':
            for id in range(len(self.prompts)):
                self.prompts[id].requires_grad = False
        elif self.prefix_or_prompt is not None:
            msg = "Incorrect prefix_or_prompt mode!"
            self.configs.logger.exception(msg)
            raise Exception(msg)
        
    def parameters(self):
        task_id = self.configs.get_task_id()
        configs_optimizer = self.configs.configs_FSCIL.optimizer
        params = []
        if self.prefix_or_prompt == 'prefix':
            if len(self.prefixes) > 0:
                params += [{'params': self.prefixes[-1], 'lr': configs_optimizer.lr_prefixes_or_prompts[task_id], 'weight_decay': configs_optimizer.weight_decay[task_id], 'type': 'delta_parameters'}]
        elif self.prefix_or_prompt == 'prompt':
            if len(self.prompts) > 0:
                params += [{'params': self.prompts[-1], 'lr': configs_optimizer.lr_prefixes_or_prompts[task_id], 'weight_decay': configs_optimizer.weight_decay[task_id], 'type': 'delta_parameters'}]
        if configs_optimizer.lr_head[task_id] > 0:
            params += dino_utils.get_params_groups(self.classifier_head, lr=configs_optimizer.lr_head[task_id])
        if not self.configs.configs_FSCIL.freeze_backbone:
            params += dino_utils.get_params_groups(self.backbone, lr=configs_optimizer.lr_backbone[task_id])
        self.backbone.zero_grad()
        return params

