# Please refer to the end of the file for references.

from __future__ import print_function
from copy import deepcopy
import torch
from torch.nn import functional as F
from dataloader.data_utils import get_datasets_and_dataloaders
from models.ViT_CCT import StochasticClassifier
from trainers.EPI import EPI, task_id_detection_mode
from utils.dino_utils import get_params_groups
from configs.Configurations import Configurations
from tqdm import tqdm
from dotmap import DotMap
from .Trainer import Trainer
from torch import Tensor as T
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler
import os
from utils.shared import MovingAverageDict, verification, prepare_optimizer, prepare_scheduler, BestModel  


class FSCIL(Trainer):
    def __init__(self, configs: Configurations):
        super().__init__(configs)
        self.prefix_or_prompt = self.configs.configs_FSCIL.configs_PEFT.prefix_or_prompt
        self.num_shots = self.configs.configs_dataset.num_shots

        self.model_EPI = EPI(configs)
        self.device = self.configs.device
        self.freeze_backbone = self.configs.configs_FSCIL.freeze_backbone
        self.prediction_net_list = configs.prediction_net_list
        
        self.configs_prediction_net = None
        self.separate_PredictionNet_for_each_task = False
        self.use_delta_parameters_for_base_task = False
        self.enable_Mahalanobis_distance = False
        
        if self.prefix_or_prompt in ['prefix' or 'prompt']:
            self.configs_prediction_net = self.configs.configs_FSCIL.PredictionNet
            self.separate_PredictionNet_for_each_task = self.configs_prediction_net.enabled and self.configs_prediction_net.separate_PredictionNet_for_each_task
            self.use_delta_parameters_for_base_task = self.configs.configs_FSCIL.use_delta_parameters_for_base_task
            self.enable_Mahalanobis_distance = self.configs.configs_FSCIL.enable_Mahalanobis_distance
    
    def learn_this_task(self) -> DotMap:
        task_id = self.configs.get_task_id()
        
        self.model_EPI.preparation_for_the_new_task()

        dataset_train, loader_train, dataset_test, loader_test = get_datasets_and_dataloaders(self.configs, self.configs.dino)

        self.training(loader_train)
        
        use_prediction_net_for_task_id = False
        
        if self.prefix_or_prompt in ['prefix', 'prompt']:
            self.task_identification_related_tasks(loader_train, loader_test)
            
            use_prediction_net_for_task_id = self.configs_prediction_net.enabled and (not self.configs_prediction_net.separate_PredictionNet_for_each_task or (self.configs_prediction_net.use_PredictionNet_for_this_task[task_id] and task_id > 0))
        
        stats = self.evaluation(loader_test,
                                use_prediction_net_for_task_id=use_prediction_net_for_task_id,     # We do not need PredictionNet when we trained the model on only one task.
                                use_prediction_net_for_embeddings=False)
        
        return stats
    
    def task_identification_related_tasks(self, loader_train, loader_test):
        task_id = self.configs.get_task_id()
        
        state_model = deepcopy(self.configs.model.state_dict())
        
        # For the PredictionNet case, we need the statistic to find the correct task-id for test samples in pseudo-labeling to select the outliers that belong to the current task.
        self.statistic_for_task_identification(loader_train,
                                               use_prediction_net_for_task_id=False,
                                               use_prediction_net_for_embeddings=False,
                                               store_or_update='store')
        
        self.configs.model.load_state_dict(state_model)
        
        if self.configs_prediction_net.enabled and (not self.configs_prediction_net.separate_PredictionNet_for_each_task or self.configs_prediction_net.use_PredictionNet_for_this_task[task_id]):
            if self.configs_prediction_net.use_pseudo_labeled_test_samples[task_id] or self.configs.configs_FSCIL.use_pseudo_labeled_samples_for_task_identification[task_id]:
                loader_train_and_pseudo_labeled_samples = \
                    self.pseudo_label_test_set_and_merge_it_with_train_set(loader_train=loader_train,
                                                                           loader_test=loader_test,
                                                                           only_samples_from_current_task=True,
                                                                           use_prediction_net_for_task_id=False,
                                                                           use_prediction_net_for_embeddings=False)
            
            self.configs.model.load_state_dict(state_model)
            
            loader_prediction_net = loader_train_and_pseudo_labeled_samples if self.configs_prediction_net.use_pseudo_labeled_test_samples[task_id] else loader_train
                
            self.train_prediction_net(loader_prediction_net, prediction_net_is_ready=False)
            
            loader_stats = loader_train_and_pseudo_labeled_samples if self.configs.configs_FSCIL.use_pseudo_labeled_samples_for_task_identification[task_id] else loader_train
        
            self.statistic_for_task_identification(loader_stats,
                                                use_prediction_net_for_task_id=False,
                                                use_prediction_net_for_embeddings=True,
                                                store_or_update='update')
            
            self.configs.model.load_state_dict(state_model)

    def update_mu(self,
                  loader_train,
                  use_prediction_net_for_embeddings: bool):
        """Updates the means in the StochasticClassifier head.
        """
        assert isinstance(self.configs.head, StochasticClassifier)
        state_model = deepcopy(self.configs.model.state_dict())

        task_id = self.configs.get_task_id()
        if task_id == 0:
            task_id_mode = task_id_detection_mode.no_prefix_or_prompt
        else:
            task_id_mode = task_id_detection_mode.previous_task_id

        embeddings_all, labels_all = self.compute_embeddings(loader_train,
                                                             task_id_mode=task_id_mode,
                                                             use_prediction_net_for_task_id=False,
                                                             use_prediction_net_for_embeddings=use_prediction_net_for_embeddings,
                                                             description="Computing the embeddings to update mu")
        
        means_for_each_class, _, labels = self.calculate_means_and_covariance(embeddings_all, labels_all, 'classes', calculate_covariance=False)

        rng = self.configs.get_this_task_range()

        for class_index, prototype in zip(labels, means_for_each_class):
            assert class_index >= rng[0] and class_index < rng[1]
            self.configs.head.mu.data[class_index] = prototype
        
        self.configs.model.load_state_dict(state_model)

    def training(self, loader_train) -> None:
        task_id = self.configs.get_task_id()
        num_epochs = self.configs.configs_FSCIL.num_epochs[task_id]
        
        if num_epochs == 0 or (self.prefix_or_prompt in ['prefix', 'prompt'] and not self.configs.has_this_task_any_delta_parameters()):   # Nothing to be trained!
            return
        
        assert (task_id == 0 and not self.use_delta_parameters_for_base_task) or num_epochs > 0
        
        if self.freeze_backbone:        # or (task_id == 0 and not self.use_delta_parameters_for_base_task)
            verification_backbone_1 = verification(self.configs, self.model_EPI.backbone.blocks[1].attn.key.weight, False, "The backbone is supposed to be frozen!")
            verification_backbone_2 = verification(self.configs, self.model_EPI.backbone.blocks[6].attn.key.weight, False, "The backbone is supposed to be frozen!")
        
        # We should train the prefixes first to be able to assign pseudo-labels to train the PredictionNet.
        if isinstance(self.configs.head, StochasticClassifier) and self.configs.configs_FSCIL.update_mu:
            self.update_mu(loader_train, use_prediction_net_for_embeddings=False)

        optimizer = prepare_optimizer(self.configs.configs_FSCIL.optimizer, self.model_EPI.parameters(), task_id, self.configs.logger)
        scheduler = prepare_scheduler(self.configs.configs_FSCIL.scheduler, optimizer)

        ma = MovingAverageDict(self.configs.configs_FSCIL.scheduler.moving_average_capacity)

        self.model_EPI.to(self.device)
        self.model_EPI.train()
        self.configs.save_current_backbone_state()
        self.configs.obtain_classifier_head_state()

        for epoch in range(num_epochs):
            correct, total = 0, 0
            total_loss = 0

            if self.freeze_backbone:
                verification_object_weights_conv_patch_embedding = verification(self.configs, self.model_EPI.backbone.patch_embedding.proj_image_size[1].weight, False, "We must not modify the frozen model!")

            # We will compare it to verify whether we can train the prefixed or not!
            if self.prefix_or_prompt == 'prefix':
                verification_object_delta_parameters = verification(self.configs, self.model_EPI.prefixes[-1], True, "Error: We can not train the prefixes!")
            elif self.prefix_or_prompt == 'prompt':
                verification_object_delta_parameters = verification(self.configs, self.model_EPI.prompts[-1], True, "Error: We can not train the prompts!")

            itr = tqdm(loader_train, desc="Iter.", dynamic_ncols=True) if self.configs.tqdm_enabled and len(loader_train) > 1 else loader_train

            self.configs.logger.info(f"Epoch: {epoch + 1}/{num_epochs}")
            
            for _, batch in enumerate(itr):
                samples = batch[0].to(self.device)
                labels = batch[1].to(self.device)
                optimizer.zero_grad()
                self.model_EPI.backbone.zero_grad()
                embeddings, task_ids = self.model_EPI(samples=samples,
                                                      task_id_mode=task_id_detection_mode.current_task_id,
                                                      use_prediction_net_for_task_id=False,
                                                      use_prediction_net_for_embeddings=False)
                
                logits = self.model_EPI.calculate_logits(embeddings, task_ids, self.configs.configs_FSCIL.evaluation.ignore_logits_for_other_tasks)
                logits_copy = logits.detach().clone()
                loss = self.model_EPI.calculate_loss(logits, labels)
                loss.backward()
                total_loss += loss.item()
                optimizer.step()
                ma.update(loss=loss.item())

                if self.freeze_backbone:
                    self.configs.reset_backbone()
                    
                if self.configs.configs_FSCIL.optimizer.lr_head[task_id] == 0:
                    self.configs.reset_classifier_head()

                pred = torch.argmax(logits_copy, dim=1)  # + index_start_current_task
                correct += torch.sum(pred == labels).item()
                total += len(labels)
            
            scheduler.step(ma['loss'].calculate())
            
            # To verify that we can train the prefixes or prompts!
            if self.prefix_or_prompt == 'prefix':
                verification_object_delta_parameters.verify(self.model_EPI.prefixes[-1])
            elif self.prefix_or_prompt == 'prompt':
                verification_object_delta_parameters.verify(self.model_EPI.prompts[-1])

            if self.freeze_backbone:
                verification_object_weights_conv_patch_embedding.verify(self.model_EPI.backbone.patch_embedding.proj_image_size[1].weight)

            self.configs.logger.info(f"Epoch {epoch + 1}/{num_epochs} Train Accuracy: {correct/total * 100: .2f}%")
            self.configs.logger.info(f"Epoch {epoch + 1}/{num_epochs} Average Loss: {total_loss/len(loader_train)} / MA Loss: {ma['loss'].calculate()}")
        
        if self.freeze_backbone:
            verification_backbone_1.verify(self.model_EPI.backbone.blocks[1].attn.key.weight)
            verification_backbone_2.verify(self.model_EPI.backbone.blocks[6].attn.key.weight)
        
    def pseudo_label_test_set_and_merge_it_with_train_set(self, loader_train, loader_test, only_samples_from_current_task: bool, use_prediction_net_for_task_id: bool, use_prediction_net_for_embeddings: bool):
        self.configs.logger.info("The pseudo-labeling phase is started ...")
        itr_train = tqdm(loader_train, desc="Obtaining the train samples and labels", dynamic_ncols=True) if self.configs.tqdm_enabled else loader_train
        self.model_EPI.eval()
        
        ignore_logits_for_other_tasks = self.configs.configs_FSCIL.evaluation.ignore_logits_for_other_tasks
        samples_train_list = []
        samples_test_list = []
        labels_train_list = []
        predictions_list = []
        task_id = self.configs.get_task_id()

        with torch.no_grad():
            for _, batch in enumerate(itr_train):
                samples = batch[0]
                labels = batch[1]
                samples_train_list.append(samples.detach().clone())
                labels_train_list.append(labels.detach().clone())
                del samples, labels

            # Test set
            if self.configs.configs_FSCIL.PredictionNet.use_pseudo_labeled_test_samples:
                itr_test = tqdm(loader_test, desc="Pseudo-labeling the test set", dynamic_ncols=True) if self.configs.tqdm_enabled else loader_test

                for _, batch in enumerate(itr_test):
                    samples = batch[0]
                    # labels = batch[1]
                    samples = samples.to(self.device)

                    if task_id == 0:        # For the base task, test set has only the base samples.
                        task_id_mode = task_id_detection_mode.current_task_id
                    else:
                        task_id_mode = task_id_detection_mode.predict_task_id
                    
                    embedding_predicted, task_ids_predicted = \
                        self.model_EPI(samples,
                                       use_prediction_net_for_task_id=use_prediction_net_for_task_id,
                                       use_prediction_net_for_embeddings=use_prediction_net_for_embeddings,
                                       # Please note that our test set contains the samples from all tasks.
                                       task_id_mode=task_id_mode
                                       )
                    
                    logits_predicted = self.model_EPI.calculate_logits(embedding_predicted, task_ids_predicted, ignore_logits_for_other_tasks)

                    labels_predicted = torch.argmax(logits_predicted, dim=1)

                    if only_samples_from_current_task:
                        assert isinstance(task_ids_predicted, T)
                        selected_indices = task_ids_predicted == task_id
                        samples = samples[selected_indices]
                        labels_predicted = labels_predicted[selected_indices]
                    
                    samples = samples.cpu()
                    labels_predicted = labels_predicted.cpu()
                    samples_test_list.append(samples)
                    predictions_list.append(labels_predicted)

        samples_all = torch.cat(samples_train_list + samples_test_list, dim=0)
        labels_and_predictions_all = torch.cat(labels_train_list + predictions_list)

        dataset_all = TensorDataset(samples_all, labels_and_predictions_all)
        loader_train_and_pseudo_labeled_test_samples = DataLoader(dataset_all, shuffle=True, batch_size=self.configs_prediction_net.batch_size_for_Pseudo_labelling, drop_last=False)
        del dataset_all
        return loader_train_and_pseudo_labeled_test_samples
    
    def train_prediction_net(self, loader, prediction_net_is_ready: bool) -> None:
        """Train the PredictionNet
        Args:
            loader (_type_): loader that contains the embeddings and labels
            prediction_net_is_ready (bool): If the PredictionNet is previously trained.
        """
        # Steps:
        # 1- Compute the prototypes for each class in the base dataset
        # 2- Find the number of outliers
        # 3- Train the PredictionNet to rectify the outliers
        self.configs.logger.info("The training of the PredictionNet is started.")
        task_id = self.configs.get_task_id()
        
        prediction_net = self.prediction_net_list[task_id if self.separate_PredictionNet_for_each_task else 0]

        prediction_net = prediction_net.to(self.device)

        embeddings, labels = self.compute_embeddings(loader,
                                                     task_id_mode=task_id_detection_mode.current_task_id,
                                                     use_prediction_net_for_task_id=prediction_net_is_ready,
                                                     use_prediction_net_for_embeddings=False,
                                                     )
        
        # To verify that we will not modify the delta parameters unintentionally!
        if self.configs.has_this_task_any_delta_parameters():
            if self.prefix_or_prompt == 'prefix':
                verification_object = verification(self.configs, self.model_EPI.prefixes[-1], False, "Error: We should not train the prefixes during the PredictionNet training!")
            elif self.prefix_or_prompt == 'prompt':
                verification_object = verification(self.configs, self.model_EPI.prompts[-1], False, "Error: We should not train the prompts during the PredictionNet training!")

        num_outliers: int = self.configs_prediction_net.num_outliers[task_id]

        outliers_list = []
        targets_list = []

        labels_space = labels.unique()
        
        for c in labels_space:
            embeds = embeddings[labels == c]
            # We calculate the distances
            prototype = embeds.mean(dim=0)      # prototype.shape = [embed_dim]
            subtraction = embeds - prototype    # subtraction.shape = [num_samples_this_class, embed_dim]
            subtraction_squared = subtraction * subtraction
            dist = subtraction_squared.sum(dim=1).sqrt()       # shape = [num_samples_this_class]
            limitation = min(num_outliers, len(labels_space), len(dist))
            if limitation == len(dist):
                self.configs.logger.warning(f"The program uses all the {len(dist)} samples in class {c} as outliers!")
            outliers_distances, outliers_indices = dist.topk(k=limitation)
            outliers = embeds[outliers_indices]
            outliers_list.append(outliers)
            
            prototype_repeated = prototype.repeat(len(outliers_indices), 1)       # We map all outliers to the same prototype.
            targets_list.append(prototype_repeated)  # We map all outliers to the same prototype.

        inputs = torch.cat(outliers_list)
        targets = torch.cat(targets_list)

        num_samples = len(targets)

        dataset_outliers = TensorDataset(inputs, targets)
        
        dataloader = DataLoader(dataset_outliers,
                                batch_size=self.configs_prediction_net.batch_size_for_PredictionNet,
                                drop_last=False,
                                sampler=SubsetRandomSampler(torch.arange(num_samples)))

        params = get_params_groups(prediction_net)
        optimizer = prepare_optimizer(self.configs_prediction_net.optimizer, params, task_id, self.configs.logger)
        ma = MovingAverageDict(self.configs_prediction_net.scheduler.moving_average_capacity)
        scheduler = prepare_scheduler(self.configs_prediction_net.scheduler, optimizer)

        num_epochs = self.configs_prediction_net.num_epochs[task_id]
        itr = range(num_epochs)

        if self.configs.tqdm_enabled:
            itr = tqdm(itr, dynamic_ncols=True)

        prediction_net.train()
        
        if self.configs_prediction_net.use_the_best_model:
            best_model = BestModel(max_or_min='min')

        for epoch in itr:
            for x, y in dataloader:
                x = x.to(self.device)
                targets = y.to(self.device)
                optimizer.zero_grad()
                outputs = prediction_net(x)
                
                if self.configs_prediction_net.loss == "squared_Euclidean_distance":
                    diff = outputs - targets
                    squared_Euclidean_distance = diff * diff
                    loss = squared_Euclidean_distance.sum()
                elif self.configs_prediction_net.loss == "MSE":
                    loss = F.mse_loss(outputs, targets)

                if self.configs_prediction_net.use_the_best_model:
                    best_model.update(prediction_net, loss.item())
                loss.backward()
                ma.update(loss=loss.item())
                optimizer.step()
            scheduler.step(ma['loss'].calculate())
            if epoch % self.configs_prediction_net.display_freq == 0 or epoch == num_epochs - 1 or epoch == 0:
                self.configs.logger.info(f"Epoch: {epoch:03}/{num_epochs}, MA loss: {ma['loss'].calculate()}")

        if self.configs_prediction_net.use_the_best_model:
            prediction_net.load_state_dict(best_model.best_state())
        prediction_net.eval()
        prediction_net.training_is_finished()

        # To verify that we have not modified the delta parameters unintentionally!
        if self.configs.has_this_task_any_delta_parameters():
            if self.prefix_or_prompt == 'prefix':
                verification_object.verify(self.model_EPI.prefixes[-1])
        
            if self.prefix_or_prompt == 'prompt':
                verification_object.verify(self.model_EPI.prompts[-1])
    
    def evaluation(self,
                   loader_test,
                   use_prediction_net_for_task_id: bool,
                   use_prediction_net_for_embeddings: bool,
                   ):
        state_model = deepcopy(self.configs.model.state_dict())
        self.model_EPI.eval()
        labels_predicted_list = []
        predictions_oracle_list = []
        labels_list = []
        task_ids_oracle_list = []
        task_ids_prediction_list = []
        task_id = self.configs.get_task_id()
        ignore_logits_for_other_tasks = self.configs.configs_FSCIL.evaluation.ignore_logits_for_other_tasks

        self.configs.logger.info(f"Evaluating the test set after task {task_id} ...")
        itr = tqdm(loader_test, desc=f"Evaluating-Task {task_id}", dynamic_ncols=True) if self.configs.tqdm_enabled else loader_test
        
        with torch.no_grad():
            for _, batch in enumerate(itr):
                samples = batch[0].to(self.device)
                labels = batch[1].to(self.device)
                
                embeddings_oracle, task_ids_oracle = \
                    self.model_EPI(samples,
                                labels=labels,
                                use_prediction_net_for_task_id=use_prediction_net_for_task_id,
                                use_prediction_net_for_embeddings=use_prediction_net_for_embeddings,
                                task_id_mode=task_id_detection_mode.oracle)
                
                logits_oracle = self.model_EPI.calculate_logits(embeddings=embeddings_oracle, task_ids=task_ids_oracle, ignore_logits_for_other_tasks=ignore_logits_for_other_tasks)

                predictions_oracle_list.append(
                    torch.argmax(logits_oracle, dim=1))
                task_ids_oracle_list.append(task_ids_oracle)
                labels_list.append(labels)

                # When we only trained the model on one task, the oracle results are enough.
                if task_id > 0:
                    embedding_predicted, task_ids_predicted = \
                        self.model_EPI(samples,
                                       use_prediction_net_for_task_id=use_prediction_net_for_task_id,
                                       use_prediction_net_for_embeddings=use_prediction_net_for_embeddings,
                                       task_id_mode=task_id_detection_mode.predict_task_id)
                    
                    logits_predicted = self.model_EPI.calculate_logits(embedding_predicted, task_ids_predicted, ignore_logits_for_other_tasks)

                    labels_predicted_current_task = torch.argmax(logits_predicted, dim=1)
                    labels_predicted_list.append(labels_predicted_current_task)
                    if self.prefix_or_prompt is None or task_ids_predicted is None:   # For the ablation study of "without_prefixes"
                        task_ids_predicted = self.model_EPI.infer_task_ids_from_labels(labels_predicted_current_task)
                    task_ids_prediction_list.append(task_ids_predicted)

            predictions_oracle = torch.cat(predictions_oracle_list)
            assert predictions_oracle.min() >= 0
            task_ids_oracle = torch.cat(task_ids_oracle_list)
            labels_all_tasks = torch.cat(labels_list)
            assert labels_all_tasks.min() >= 0
            assert task_id > 0 or predictions_oracle.max() < self.configs.configs_dataset.num_base_classes
            
            stats = DotMap()
            self.configs.model.load_state_dict(state_model)
            
            stats.number_of_samples = len(labels_all_tasks)
            stats.correct.oracle = torch.sum(predictions_oracle == labels_all_tasks).item()
            stats.accuracy.oracle = stats.correct.oracle / stats.number_of_samples

            self.configs.logger.info(f"Evaluation Accuracy (oracle) after task {task_id}: {stats.accuracy.oracle * 100.0: .2f}%")

            if task_id > 0:
                labels_predicted = torch.cat(labels_predicted_list)
                assert labels_predicted.min() >= 0
                assert predictions_oracle.max() > self.configs.configs_dataset.num_base_classes
                task_ids_predicted = torch.cat(task_ids_prediction_list)
                stats.correct.predictions = torch.sum(labels_predicted == labels_all_tasks).item()
                stats.accuracy.predictions = stats.correct.predictions / stats.number_of_samples
                stats.correct_task_ids = torch.sum(task_ids_predicted == task_ids_oracle).item()
                stats.accuracy.task_id_detection = stats.correct_task_ids / stats.number_of_samples
                self.configs.logger.info(f"Evaluation Accuracy after task {task_id}: {stats.accuracy.predictions * 100.0: .2f}%")
                self.configs.logger.info(f"Accuracy of task-id detection after task {task_id}: {stats.accuracy.task_id_detection * 100.0: .2f}%")

            for t_id in range(task_id + 1):
                stats.total[t_id] = torch.sum(task_ids_oracle == t_id).item()
                stats.correct.oracle_tasks[t_id] = torch.sum(torch.logical_and(predictions_oracle == labels_all_tasks, task_ids_oracle == t_id)).item()
                stats.accuracy.oracle_tasks[t_id] = stats.correct.oracle_tasks[t_id] / stats.total[t_id]
                self.configs.logger.info(f"Accuracy (Oracle) for task {t_id} = {stats.accuracy.oracle_tasks[t_id] * 100.0: .2f}%")
                if task_id > 0:
                    stats.correct.predictions_for_task[t_id] = torch.sum(torch.logical_and(labels_predicted == labels_all_tasks, task_ids_oracle == t_id)).item()
                    stats.accuracy.predictions_for_task[t_id] = stats.correct.predictions_for_task[t_id] / stats.total[t_id]
                    self.configs.logger.info(f"Accuracy of task {t_id} = {stats.accuracy.predictions_for_task[t_id] * 100.0: .2f}%")
            return stats

    def statistic_for_task_identification(self,
                                          loader,
                                          use_prediction_net_for_task_id: bool,
                                          use_prediction_net_for_embeddings: bool,
                                          store_or_update: str) -> None:
        assert store_or_update in ['store', 'update']
        # This method, first, calculates the means and covariances. Second, it stores and accumulates the statistics in the EPI object.
        self.configs.logger.info("Statistics for task identification ...")
        temp_str = "calibrated " if use_prediction_net_for_embeddings else ""
        desc = f"Computing the {temp_str}embeddings"
        
        res = self.compute_embeddings(loader,
                                      task_id_mode=task_id_detection_mode.current_task_id,
                                      use_prediction_net_for_task_id=use_prediction_net_for_task_id,
                                      use_prediction_net_for_embeddings=use_prediction_net_for_embeddings,
                                      description=desc)
        
        embeddings_current_task, labels_current_task = res
        
        means_for_each_class, covariance, labels_space = \
            self.calculate_means_and_covariance(embeddings_all=embeddings_current_task,
                                                tasks_or_classes=self.configs.configs_FSCIL.tasks_or_classes_for_Mahalanobis_distance_calculations,
                                                labels=labels_current_task,
                                                calculate_covariance=self.enable_Mahalanobis_distance)
            
        if store_or_update == 'store':
            self.model_EPI.statistics_for_task_identification.store_and_accumulate_the_statistics_for_this_task(means_for_each_class, covariance, labels_space)
        else:
            self.model_EPI.statistics_for_task_identification.update_the_statistics_for_the_last_task(means_for_each_class, covariance, labels_space)
    
    def compute_embeddings(self,
                           loader,
                           task_id_mode: task_id_detection_mode,
                           use_prediction_net_for_task_id: bool,
                           use_prediction_net_for_embeddings: bool,
                           return_labels=True,
                           description="Computing the embeddings"):
        self.model_EPI.eval()
        itr = tqdm(loader, desc=description, dynamic_ncols=True) if self.configs.tqdm_enabled else loader

        with torch.no_grad():
            embeddings_list = []
            labels_list = []
            for _, batch in enumerate(itr):
                samples = batch[0]
                labels = batch[1]
                samples = samples.to(self.device)

                embeddings, task_ids = self.model_EPI(samples=samples,
                                                      task_id_mode=task_id_mode,
                                                      use_prediction_net_for_task_id=use_prediction_net_for_task_id,
                                                      use_prediction_net_for_embeddings=use_prediction_net_for_embeddings
                                                      )
                
                # Each row of prelogits is the average of embeddings over tokens for each sample.
                embeddings_list.extend(embeddings.tolist())
                if return_labels:
                    labels_list.extend(labels.tolist())

        embeddings_all = torch.tensor(embeddings_list, device=self.device)
        
        if return_labels:
            labels_all = torch.tensor(labels_list)
            return embeddings_all, labels_all
        else:
            return embeddings_all, None

    def calculate_means_and_covariance(self, embeddings_all: T, labels: T, tasks_or_classes: str, calculate_covariance: bool = True):
        assert tasks_or_classes in ['tasks', 'classes']
        if tasks_or_classes == 'classes':
            labels_space = labels.unique()

            means_for_each_class_list = []
            cov_over_classes_list = []
            
            for c in labels_space:
                embeds = embeddings_all[labels == c]
                mean = embeds.mean(dim=0)
                means_for_each_class_list.append(mean)
                if calculate_covariance:
                    covariance = torch.cov((embeds - mean).T)
                    cov_over_classes_list.append(covariance)

            means_for_each_class = torch.stack(means_for_each_class_list)
            
            if calculate_covariance:
                covariance = torch.stack(cov_over_classes_list).mean(dim=0)
                return means_for_each_class, covariance, labels_space
            else:
                return means_for_each_class, None, labels_space
        else:       # self.configs.configs_FSCIL.tasks_or_classes_for_Mahalanobis_distance_calculations == 'tasks'
            embeds = embeddings_all
            mean = embeds.mean(dim=0)
            if calculate_covariance:
                covariance = torch.cov((embeds - mean).T)

            task_id = torch.tensor(self.configs.get_task_id(), device=self.device)
            if calculate_covariance:
                covariance = covariance.mean(dim=0)
                return mean, covariance, task_id
            else:
                return mean, None, task_id


# @inproceedings{wang2023rehearsal,
#   title={Rehearsal-free Continual Language Learning via Efficient Parameter Isolation},
#   author={Wang, Zhicheng and Liu, Yufang and Ji, Tao and Wang, Xiaoling and Wu, Yuanbin and Jiang, Congcong and Chao, Ye and Han, Zhencong and Wang, Ling and Shao, Xu and others},
#   booktitle={Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
#   pages={10933--10946},
#   year={2023}
# } -> https://github.com/Dicer-Zz/EPI

# @inproceedings{Liu2019PrototypeRF,
#   title={Prototype Rectification for Few-Shot Learning},
#   author={Jinlu Liu and Liang Song and Yongqiang Qin},
#   booktitle={European Conference on Computer Vision},
#   year={2019}
# }
