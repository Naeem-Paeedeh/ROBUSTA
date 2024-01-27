# You don't need to set anything here. We set the arguments in TOML files.
# We used some codes from S3C(https://github.com/JAYATEJAK/S3C) and https://github.com/moukamisama/F2M

import sys
from utils import dino_utils
import torch
from torch import nn
import numpy as np
from configs.configs_model import ConfigurationModel, string_to_model_config
from models.ViT_CCT import CosineClassifier, StochasticClassifier, ViT_CCT, PositionalEmbeddingType, DINOHead
import random
import os
import logging
from copy import deepcopy
import gc
from utils.shared import get_env_info, get_root_logger, get_time_str, Box2str, convert_none_string_to_None, parse_str_bool, save_a_list_to_a_text_file
import tomli
from box import Box
from models.PredictionNet import PredictionNet


class Configurations:
    """This class keep all settings and states. Therefore, we just pass it to many function to convey the arguments,
    and each method uses what it requires for its operations.
    """
    def __init__(self, args):
        super().__init__()

        self.experiment_description = ""
        self.phase = ""
        self.device: torch.device = None
        self.seed = 0
        self.is_incremental = False
        self.logger = None

        self.tqdm_enabled = False
        self.resume = False
        self.time_str = get_time_str()

        self.configs_arch = None
        self.configs_dataset = None
        self.configs_train = None
        self.configs_dino = None
        self.configs_logger = None
        self.configs_save = None
        self.configs_FSCIL = None
        self.augmentations = None
        self.transformations = None

        self.image_size: int = 0
        self.in_channels: int = 0

        # FSCIL settings:
        self.batch_size_base: int = 0
        self.batch_size_test: int = 0
        self.batch_size_new: int = 0
        self.batch_size_fine_tuning: int = 0
        self._task_id = 0
        self.prototypes: torch.Tensor = None

        # All settings in the arguments and TOML file can be accessed by settings.
        # {
        args_dict = vars(args)

        try:
            with open(args.settings_file, mode="rb") as toml_file:
                args_toml = tomli.load(toml_file)
        except:
            msg = f"Error: The setting file \"{args.settings_file}\" can not be loaded!"
            # self.logger.exception(msg)
            raise Exception(msg)

        all_args = args_dict | args_toml

        for k, v in all_args.items():
            if isinstance(v, dict):
                setattr(self, k, convert_none_string_to_None(Box(v)))
            else:
                setattr(self, k, v)
        
        # The settings of the learning rate scheduler
        # We have two different optimizers for supervised and self-supervised learning.
        self.optimizer = None
        self.optimizer_dino = None
        self.scheduler_dino_lr_backbone = None
        self.scheduler_dino_lr_head = None
        # Weight decay scheduler
        self.wd_schedule_dino = None
        self.momentum_schedule_dino = None
        # }
        
        # We want to resume the process for FSCIL easily. Therefore, we just save date for time string in the root directory name.
        self.configs_save.root = os.path.expanduser(self.configs_save.root)
        
        os.makedirs(self.configs_save.root, exist_ok=True)
        self.prepare_logger()

        # To make save files for each experiment unique!
        self.configs_save.output_file += f",start_time={self.time_str}"

        self.set_seed()

        num_incremental_tasks = self.configs_dataset.num_tasks - 1
        total_incremental_classes = self.configs_dataset.total_classes - self.configs_dataset.num_base_classes
        assert self.configs_dataset.num_tasks == 1 or total_incremental_classes % num_incremental_tasks == 0
        
        self.configs_dataset.num_ways = total_incremental_classes // num_incremental_tasks if self.configs_dataset.num_tasks > 1 else 0
        
        assert self.configs_dataset.dataset_name in ['mini_imagenet', 'cifar100', 'cub200', "imagenet"]

        self.directory_permutation_files = os.path.join('data', 'index_list', self.configs_dataset.dataset_name, f"num_base_classes={self.configs_dataset.num_base_classes},num_tasks={self.configs_dataset.num_tasks},num_shots={self.configs_dataset.num_shots},seed={self.seed}")
        
        self.class_permutation = None
        
        self.load_class_permutation_file()

        # For resuming the experiment
        self.epoch = 0
        self.iteration = 0

        # We calculate the relative path of the files in the specified save directory.

        # Configuration of the model
        self.configs_model: ConfigurationModel = string_to_model_config[self.configs_arch.model_type]
        self.configs_model.use_BatchNorm = self.configs_arch.use_BatchNorm
        self.configs_model.logger = self.logger

        self.is_cuda_available()

        self._logger_level_saved = None
      
        self.model = None
        self.teacher = None

        self._state_model = None
        self._state_teacher = None
        self._state_head = None

        self.history_FSCIL = {}

        assert self.phase in ["self_supervised_learning", "supervised_learning", "incremental_learning", "fine_tuning", "supervised_learning_with_prefixes"]
        
        if self.phase == "incremental_learning":
            self.is_incremental = True

        self.dino: bool = self.phase == 'self_supervised_learning'    # and self.configs_train.gamma_self_supervised > 0.0

        self.dino_transforms = None

        self.model = self.initialize_model(use_wrapper_for_dino=self.dino, use_BatchNorm_in_dino_head=self.configs_dino.use_BatchNorm_in_dino_head if self.dino else None)
        self._state_model = deepcopy(self.model.state_dict())
        
        if self.dino:
            # Please note that we had set use_BatchNorm_in_dino_head of the teacher to false in the self-supervised learning phase of the Mini-ImageNet and CIFAR-100 experiments.
            self.teacher = self.initialize_model(use_wrapper_for_dino=self.dino, use_BatchNorm_in_dino_head=self.configs_dino.use_BatchNorm_in_dino_head)
            self._state_teacher = deepcopy(self.model.state_dict())
            self.teacher.load_state_dict(self._state_teacher)
            self.dino_transforms = dino_utils.DataAugmentationDINO(
                global_crops_scale=self.configs_dino.global_crops_scale,
                local_crops_scale=self.configs_dino.local_crops_scale,
                local_crops_number=self.configs_dino.local_crops_number,
                image_size=self.image_size,
                local_patch_size=self.configs_dino.local_patch_size,
                dataset_name=self.configs_dataset.dataset_name,
                enable_normalize_transform=self.configs_dino.enable_normalize_transform
            )
        
        self.head = None
        
        if self.phase != 'self_supervised_learning' and hasattr(self.configs_arch, 'classifer_head_type'):
            head_type = self.configs_arch.classifer_head_type
            dim_input = self.configs_model.embed_dim
            if head_type == 'Stochastic':
                self.head = StochasticClassifier(self.device,
                                                 dim_input,
                                                 self.configs_dataset.total_classes,
                                                 self.configs_arch.temperature_stochastic_classifier)
            elif head_type == 'Cosine':
                self.head = CosineClassifier(self.device,
                                             dim_input,
                                             self.configs_dataset.total_classes,
                                             self.configs_arch.temperature_cosine_classifier)
            elif head_type == 'Linear':
                self.head = torch.nn.Linear(dim_input,
                                            self.configs_dataset.total_classes,
                                            device=self.device)
            else:
                msg = ""
                self.logger.exception(msg)
                raise Exception(msg)
            
        self.prefixes_base_task = None
        
        if self.phase == 'supervised_learning_with_prefixes':
            self.initialize_prefixes_for_base_task()
        
        assert self.phase != 'incremental_learning' or self.configs_FSCIL.start_from_task == 0  # We must start from task 0 to be able to use the PredictionNet!
        
        self.load()

        self.prediction_net_list: list[PredictionNet] = []

        gc.collect()
        
        self.label_to_task_id_dictionary = []
        self.prepare_label_to_task_id_dictionary()

        if self.phase == "incremental_learning":
            
            num_tasks = self.configs_dataset.num_tasks
            assert len(self.configs_FSCIL.num_epochs) == num_tasks
            assert len(self.configs_FSCIL.optimizer.weight_decay) == num_tasks
            assert len(self.configs_FSCIL.optimizer.lr_head) == num_tasks
            assert len(self.configs_FSCIL.optimizer.lr_backbone) == num_tasks
            assert len(self.configs_FSCIL.optimizer.lr) == num_tasks
            # We can not copy the previous prefix content if the size of the current prefix is different.
            
            if self.configs_FSCIL.configs_PEFT.prefix_or_prompt in ['prefix', 'prompt']:
                assert self.configs_FSCIL.configs_PEFT.fusion_mode in ['last', 'random', 'mean', 'zeros']
                assert len(self.configs_FSCIL.use_pseudo_labeled_samples_for_task_identification) == num_tasks
                assert len(self.configs_FSCIL.configs_PEFT.number_of_layers_for_prefixes) == num_tasks
                assert self.configs_FSCIL.tasks_or_classes_for_Mahalanobis_distance_calculations in ['tasks', 'classes']
                assert len(self.configs_FSCIL.optimizer.lr_prefixes_or_prompts) == num_tasks
                
                configs_prediction_net = self.configs_FSCIL.PredictionNet
                if configs_prediction_net.enabled:
                    assert len(configs_prediction_net.num_outliers) == num_tasks
                    assert len(configs_prediction_net.use_pseudo_labeled_test_samples) == num_tasks
                    assert len(configs_prediction_net.num_epochs) == num_tasks
                    assert len(configs_prediction_net.optimizer.lr) == num_tasks
                    assert configs_prediction_net.loss in ["squared_Euclidean_distance", "MSE"]
                    assert 'use_PredictionNet_for_this_task' not in configs_prediction_net or len(configs_prediction_net.use_PredictionNet_for_this_task) == num_tasks

        self.print_arguments()

    def print_arguments(self):
        attributes = self.__dict__
        ignore_set = {'args', 'log_file', 'history_FSCIL', 'label_to_task_id_dictionary'}
        useful_types = {int, float, str, list, bool, tuple, dict, Box, torch.device}

        message = "The given arguments"
        temp = 80 - 2 - len(message) // 2   # :)
 
        self.logger.info('-' * temp + ' ' + message + ' ' + '-' * temp)
        
        wait_list = []

        for name in attributes.keys():
            value = getattr(self, name)
            type_attr = type(value)
            if name.startswith("_") or name in ignore_set or type_attr not in useful_types or value is None:
                continue

            if name == 'epoch':
                self.logger.info("The network was previously trained for %d epochs." % self.epoch)
            elif name == 'iteration':
                self.logger.info("The network was previously trained for %d iterations." % self.iteration)
            elif name == 'device':
                self.logger.info("Device = \"%s\"", str(self.device))
            elif type_attr == Box:
                wait_list.append((name, value))
            elif type_attr != str:
                self.logger.info("%s = %s" % (name, str(getattr(self, name))))
            else:
                self.logger.info('%s = "%s"' % (name, getattr(self, name)))

        # We show these settings at last.
        for name, value in wait_list:
            temp = f"\n{name}: {Box2str(value)}"
            self.logger.info(temp)
        
        self.logger.info('-' * 80)
    
    def prepare_logger(self):
        if self.phase == 'incremental_learning':
            logger_name = 'Inc_Learning'
        elif self.phase == 'supervised_learning':
            logger_name = 'BaseTask_Sup'
        elif self.phase == 'self_supervised_learning':
            logger_name = 'BaseTask_Self-Sup'
        else:
            logger_name = 'Unknown phase'

        logger_dir = self.configs_save.root

        os.makedirs(logger_dir, exist_ok=True)
        self.configs_logger.log_file = os.path.join(logger_dir, f"Time={self.time_str},Desc={self.experiment_description},seed={self.seed}.log")

        self.logger = get_root_logger(logger_name=logger_name, log_level=logging.INFO, log_file=self.configs_logger.log_file)

        self.logger.info(get_env_info())

    def initialize_prefixes_for_base_task(self):
        number_of_layers_for_prefixes = self.configs_train.configs_PEFT.number_of_layers_for_prefixes
        
        if number_of_layers_for_prefixes == -1:
            number_of_layers_for_prefixes = self.configs_model.num_layers
        
        prefixes_base_task_tensor = torch.randn(
            (number_of_layers_for_prefixes,
             2,
             self.configs_model.transformer_num_heads,
             self.configs_train.prefix_seq_length,
             self.configs_model.head_dim), device=self.device, requires_grad=True)
        
        self.prefixes_base_task = nn.Parameter(prefixes_base_task_tensor, requires_grad=True)
    
    def load_class_permutation_file(self):
        """This method loads the class_permutation if it saved it before. Otherwise, it creates a new file at the first phase of each experiment.
        # dataset_name = self.configs_dataset.dataset_name
        """

        os.makedirs(self.directory_permutation_files, exist_ok=True)

        perm_file_name = os.path.join(self.directory_permutation_files, "class_permutation.txt")

        if os.path.exists(perm_file_name):
            with open(perm_file_name, "r") as f:
                perm_str = f.read()
                classes = perm_str.split(',')
                self.class_permutation = torch.tensor([int(label) for label in classes])

            self.logger.info(f"class_permutation is loaded from the permutation file \"{perm_file_name}\".")
        else:
            if self.configs_FSCIL.randomize_selected_classes:
                self.class_permutation = torch.randperm(self.configs_dataset.total_classes)
            else:
                self.class_permutation = torch.arange(self.configs_dataset.total_classes)
            save_a_list_to_a_text_file(self.class_permutation.tolist(), perm_file_name)
            self.logger.info(f"Permutation file \"{perm_file_name}\" is created.")
            
    def is_cuda_available(self):
        """
        It checks if the cuda is available, then devices for different configurations to the result.
        :return:
        """
        self.device = torch.device(self.device) if torch.cuda.is_available() else 'cpu'

        return torch.cuda.is_available()

    def initialize_model(self, use_wrapper_for_dino: bool, use_BatchNorm_in_dino_head=None, pretrained_model_state=None, strict=True):
        if self.configs_arch.PositionalEmbeddingType == "Learnable":
            pos_emb_type = PositionalEmbeddingType.Learnable
        elif self.configs_arch.PositionalEmbeddingType == "SinCos":
            pos_emb_type = PositionalEmbeddingType.SinCos
        else:
            msg = f"Error: configs_arch.PositionalEmbeddingType={self.configs_arch.PositionalEmbeddingType} is incorrect!"
            self.logger.exception(msg)
            raise Exception(msg)

        model = ViT_CCT(configs_model=self.configs_model,
                        use_proj_dino=self.dino,
                        use_BatchNorm_for_patch_embeddings=self.configs_arch.use_BatchNorm_for_patch_embeddings,
                        use_BatchNorm_for_patch_embeddings_for_local_patches=self.configs_arch.use_BatchNorm_for_patch_embeddings_for_local_patches if self.dino else False,
                        image_size=self.image_size,
                        local_patch_size=self.configs_dino.local_patch_size if self.dino else -1,
                        in_channels=self.in_channels,
                        pos_embedding_type=pos_emb_type,
                        )
        
        if pretrained_model_state is not None:
            # For DINO, we should wrap the pretrained model here!
            msg = model.load_state_dict(pretrained_model_state, strict=strict)
            print(f'--> The weights of the pre-trained model is loaded with message: {msg}')

        if use_wrapper_for_dino:
            assert use_BatchNorm_in_dino_head is not None
            model = dino_utils.MultiCropWrapper(model,
                                                DINOHead(
                                                    self.configs_model.embed_dim,
                                                    self.configs_dino.out_dim,
                                                    use_bn=False,
                                                    norm_last_layer=use_BatchNorm_in_dino_head,
                                                )
                                                )
        return model

    def get_the_model(self):
        self.model.to(self.device)
        return self.model
    
    def get_the_classifier_head(self):
        self.head.to(self.device)
        return self.head

    def get_the_teacher(self):
        self.teacher.to(self.device)
        return self.teacher
    
    # Set seed for reproducibility
    def set_seed(self, seed=None):
        """Sets the seed of random number generators to the predefined seed number for reproducibility.
        """
        if seed is None:
            seed = self.seed
        # torch.use_deterministic_algorithms(True)
        np.random.seed(seed)
        random.seed(seed)
        torch.random.manual_seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def save(self,
             epoch: int,
             iteration: int,
             snapshot=False,
             is_this_the_best_model=False):
        self.epoch = epoch
        self.iteration = iteration

        state = {'model': deepcopy(self.model.state_dict()),
                 'model_type': self.configs_arch.model_type}

        if self.teacher is not None:
            state['teacher'] = deepcopy(self.teacher.state_dict())
        state['epoch'] = epoch
        state['iteration'] = iteration

        if self.head is not None:
            self._state_head = deepcopy(self.head.state_dict())
            state['head'] = self._state_head

        state['phase'] = self.phase

        state['task_id'] = self._task_id

        state['history_FSCIL'] = self.history_FSCIL
        
        if self.prefixes_base_task is not None:
            state['prefixes_base_task'] = self.prefixes_base_task

        if not os.path.exists(self.configs_save.root):
            os.makedirs(self.configs_save.root, exist_ok=True)

        path = self.get_save_file_name_prefix() + f",seed={self.seed}"
        
        path_last_without_task_id = path + '-Last.pt'
        torch.save(state, path_last_without_task_id)
        self.logger.info(f'\nThe last model is saved in "{path_last_without_task_id}"! <--------')

        path_last_with_task_id = path + f",Task-ID={self._task_id}" + '-Last.pt'
        torch.save(state, path_last_with_task_id)
        self.logger.info(f'\nThe model is saved in "{path_last_with_task_id}"! <--------')

        if snapshot:
            if epoch > 0:
                path += f"-Epoch={self.epoch}"
            elif iteration > 0:
                path += f"-Iter={self.iteration}"

            path += '.pt'

            if path != self.configs_save.output_file:
                torch.save(state, path)
                self.logger.info(f'\nA snapshot is saved to "{path}" <--------')
        elif is_this_the_best_model:
            path += '-Best_Model.pt'
            torch.save(state, path)
            self.logger.info(f'\nThis model is saved as the best model in "{path}"! <--------')

    def get_save_file_name_prefix(self,):
        path_main = os.path.join(self.configs_save.root, self.configs_save.output_file)
        return path_main
    
    def load(self):
        """
        Loads the last state from the disk. If the process finished before, it will return true. Some the states of
        some PyTorch modules are saved in the variables to be called later
        :param key: If we need just the value for a specific key. We skip from loading the other parts.
        :return:
        """
        file_path = self.configs_save.input_file
        
        # It is not working. But, it does not affect the process.
        def remove_proj_dino_from_state_dict(state_model):
            removal_list = []
            for k in state_model:
                if 'proj_dino' in k:
                    removal_list.append(k)
            for k in removal_list:
                state_model.pop(k)
        
        def remove_head_dino_from_state_dict(state_model):
            removal_list = []
            for k in state_model:
                if k.startswith('head.'):
                    removal_list.append(k)
            for k in removal_list:
                state_model.pop(k)

        if os.path.exists(file_path):
            state = torch.load(file_path, map_location="cpu")

            # If the phase has not changed, we resume the process, otherwise we ignore some settings.
            same_phase = (self.phase == state['phase'])

            epoch_in_the_save_file = 0
            iteration_in_the_save_file = 0
            if 'epoch' in state:
                epoch_in_the_save_file = state['epoch']
            if 'iteration' in state:
                iteration_in_the_save_file = state['iteration']
                
            if not same_phase or self.phase == 'incremental_learning' or not self.resume:
                self.epoch = 0
                self.iteration = 0

            self.logger.info("The network was trained for %d epochs, %d iterations in phase %s", epoch_in_the_save_file, iteration_in_the_save_file, state['phase'])

            if same_phase:
                self.epoch = epoch_in_the_save_file
                self.iteration = iteration_in_the_save_file

            if 'model' in state:
                self.model_type = state['model_type']
                if self.dino:       # If the current phase is self_supervised_learning
                    if state['phase'] == 'self_supervised_learning':  # If the past phase was self_supervised_learning (resuming)
                        self._state_model = state['model']
                        self.model.load_state_dict(self._state_model)

                        # We also have the teacher model's weights in the saved file
                        self._state_teacher = state['teacher']
                    else:
                        # If the past phase was not self_supervised_learning.
                        # It means that we trained it with another method, and we do not resume self-supervised learning.
                        # We should wrap the model and add a head for DINO
                        # First, we remove the proj_dino from the state dictionary if it exists in the model, because we may want to use a different local_patch_size
                        self.logger.info('We remove the useless proj_dino from the loaded model!')
                        remove_proj_dino_from_state_dict(state['model'])
                        self.model = self.initialize_model(use_wrapper_for_dino=True,
                                                        use_BatchNorm_in_dino_head=self.configs_dino.use_BatchNorm_in_dino_head,
                                                        pretrained_model_state=state['model'],
                                                        strict=False)
                        self.model.zero_grad()      # It may be redundant!
                        self._state_model = deepcopy(self.model.state_dict())
                        self._state_teacher = deepcopy(self.model.state_dict())
                    # Please note that we had set use_BatchNorm_in_dino_head of the teacher to false in the self-supervised learning phase of the Mini-ImageNet and CIFAR-100 experiments.
                    self.teacher = self.initialize_model(use_wrapper_for_dino=True, use_BatchNorm_in_dino_head=self.configs_dino.use_BatchNorm_in_dino_head)
                    self.teacher.load_state_dict(self._state_teacher)
                else:   # If the current phase is NOT self_supervised_learning
                    if 'teacher' in state and state['phase'] == 'self_supervised_learning':  # If we trained the model with DINO in the previous step
                        # When we do not continue the training of DINO, we load the pretrained weights of the teacher.
                        remove_proj_dino_from_state_dict(state['teacher'])
                        remove_head_dino_from_state_dict(state['teacher'])
                        state_teacher_dict = state['teacher']
                        # remove `module.` prefix
                        state_teacher_dict = {k.replace("module.", ""): v for k, v in state_teacher_dict.items()}
                        # remove `backbone.` prefix induced by multicrop wrapper
                        state_teacher_dict = {k.replace("backbone.", ""): v for k, v in state_teacher_dict.items()}
                        msg = self.model.load_state_dict(state_teacher_dict, strict=True)
                        self.model.zero_grad()      # It may be redundant!
                        self._state_model = deepcopy(self.model.state_dict())
                        print('-' * 80)
                        print(f'--> The weights of the teacher model is loaded with message: {msg}')
                    else:   # We are loading a saved model, which is not trained with DINO.
                        remove_proj_dino_from_state_dict(state['model'])
                        self._state_model = state['model']
                        self.model.load_state_dict(self._state_model)
                        # We load the teacher again when we resume the training with DINO
                    # We will not save the teacher anymore.
                    self._state_teacher = None
                    self.teacher = None

            if self.phase not in ['self_supervised_learning'] and 'head' in state:
                self._state_head = state['head']
                if 'mu' in self._state_head.keys() and self._state_head['mu'].shape[0] == self.configs_dataset.total_classes:
                    self.head.load_state_dict(self._state_head)
                    self.head = self.head.to(self.device)
                    self.logger.info("We have loaded the head parameters from the saved file.")
                else:
                    self.logger.warning('We ignored the head parameters because the previous head was trained on a different datasets.')

            if 'prefixes_base_task' in state:
                self.initialize_prefixes_for_base_task()
                self.prefixes_base_task.load_state_dict(state['prefixes_base_task'])
            
            self.logger.info("We start from epoch %d, iteration %d", self.epoch, self.iteration)

            if 'task_id' in state:
                self._task_id = state['task_id']

            if 'history_FSCIL' in state:
                self.history_FSCIL = state['history_FSCIL']

            if self.phase == 'incremental_learning':
                self.teacher = None
                self._state_teacher = None

            self.logger.info("File \"%s\" is loaded", file_path)
        else:
            self.logger.warning('No save file is loaded!')
            if file_path != "":
                ans = parse_str_bool(input("Do you want to train a model from scratch? "))
                if not ans:
                    sys.exit(0)

            dino = self.phase == 'self_supervised_learning'  # or 'teacher' in state
            self.logger.warning("We are starting to train the model from scratch!")
            if self.configs_dino is not None:
                self.model = self.initialize_model(dino, self.configs_dino.use_BatchNorm_in_dino_head if self.dino else None)
            else:
                self.model = self.initialize_model(dino)
            self.model.zero_grad()      # It may be redundant!
            self._state_model = deepcopy(self.model.state_dict())
            if dino:
                self.teacher = self.initialize_model(dino, False)
                self._state_teacher = deepcopy(self.model.state_dict())
                self.teacher.load_state_dict(self._state_teacher)

    def collect_garbage(self):
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.set_device(self.device)
            torch.cuda.empty_cache()
    
    def reset_backbone(self):
        if self._state_model is not None:
            self.model.load_state_dict(self._state_model, strict=True)
            self.model.zero_grad()
        else:
            msg = "Error: The model was not loaded!"
            self.logger.exception(msg)
            raise Exception(msg)
        
    def save_current_backbone_state(self):
        self._state_model = deepcopy(self.model.state_dict())
        
    def reset_classifier_head(self):
        if self._state_head is not None:
            self.head.load_state_dict(self._state_head, strict=True)
        else:
            msg = "Error: The classifier head was not loaded!"
            self.logger.exception(msg)
            raise Exception(msg)
        
    def obtain_classifier_head_state(self):
        self._state_head = deepcopy(self.head.state_dict())
        return self._state_head
    
    def load_classifier_head(self, state):
        self._state_head = deepcopy(state)
        self.head.load_state_dict(self._state_head)
        
    def set_task_id(self, task_id: int):
        self._task_id = task_id

    def get_task_id(self) -> int:
        return self._task_id
    
    def get_learned_classes_and_current_task_range(self, task_id: int = None) -> tuple[int, int]:
        """It calculates the valid range of consecutive indices of the classes which our model learned and current task.

        Args:
            task_id (int, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        if task_id is None:
            task_id = self._task_id
        assert task_id >= 0
        return 0, self.configs_dataset.num_base_classes + task_id * self.configs_dataset.num_ways
        
    def get_this_task_range(self, task_id: int = None) -> tuple[int, int]:
        """It calculates the valid range of consecutive indices for the task

        Args:
            task_id (int): Task ID. If it is not given, it consider the self._task_id for computations.

        Returns:
            tuple: index_start, index_end
        """
        if task_id is None:
            task_id = self._task_id
        assert task_id >= 0
        index_start = self.configs_dataset.num_base_classes + (task_id - 1) * self.configs_dataset.num_ways if task_id > 0 else 0

        index_end = self.configs_dataset.num_base_classes + task_id * self.configs_dataset.num_ways
        return index_start, index_end
    
    def has_this_task_any_delta_parameters(self, task_id: int = None) -> bool:
        if task_id is None:
            task_id = self._task_id
        return (self.configs_FSCIL.use_delta_parameters_for_base_task or task_id > 0) and self.configs_FSCIL.configs_PEFT.prefix_or_prompt in ['prefix', 'prompt']

    def get_selected_classes(self, train_or_test: str, task_id: int = None):
        assert train_or_test in ["train", "test"]
        if train_or_test == "train":
            rng = self.get_this_task_range(task_id)
        else:
            rng = self.get_learned_classes_and_current_task_range(task_id)

        return self.class_permutation[rng[0]:rng[1]]
    
    def prepare_label_to_task_id_dictionary(self):
        self.label_to_task_id_dictionary = [0] * self.configs_dataset.total_classes
        
        for task_id in range(self.configs_dataset.num_tasks):
            rng = self.get_this_task_range(task_id)
            for label in range(rng[0], rng[1]):
                self.label_to_task_id_dictionary[label] = task_id
        pass
