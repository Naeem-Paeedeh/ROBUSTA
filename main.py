#!/usr/bin/env python3

# Please refer to the end of the file for references that we used.

from __future__ import absolute_import, division, print_function
import os
import math
import sys
import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from dataloader.data_utils import get_dataloader_train, get_datasets_and_dataloaders
from models.MLP_V import MLP_V
from trainers.Supervised_learning_with_prefixes import Supervised_learning_with_prefixes
from utils.shared import AveragerDict, parse_arguments, Stopwatch, MovingAverageDict, BestResult, prepare_optimizer, prepare_scheduler, print_estimated_remaining_time, verification
from configs.Configurations import Configurations
from utils import dino_utils
from copy import deepcopy
from torch.optim import lr_scheduler
import torch.multiprocessing
import torch.nn.functional as F
from dotmap import DotMap
from torch.utils.data import TensorDataset, DataLoader  # , Subset
from models.ViT_CCT import StochasticClassifier, freeze_the_first_layers
import gc
from trainers.FSCIL import FSCIL
import traceback


# We used the code from https://github.com/facebookresearch/dino
def train_one_epoch(configs: Configurations, student, teacher, dino_loss, data_loader):
    epoch = configs.epoch
    optimizer_dino = configs.optimizer_dino
    scheduler_dino_lr_backbone = configs.scheduler_dino_lr_backbone
    scheduler_dino_lr_head = configs.scheduler_dino_lr_head
    wd_schedule_dino = configs.wd_schedule_dino
    momentum_schedule_dino = configs.momentum_schedule_dino

    metric_logger = dino_utils.MetricLogger(delimiter="  ")
    header = f"Epoch: [{epoch}/{configs.configs_dino.num_epochs}]"
    
    coef_loss_dino = configs.configs_dino.coef_loss_dino
    coef_loss_ce = configs.configs_dino.coef_loss_ce

    for it, data in enumerate(metric_logger.log_every(data_loader, 50, header), 0):
        images = data[0]
        labels = data[1].long().to(configs.device)

        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer_dino.param_groups):
            if param_group['type'] == 'backbone':
                param_group["lr"] = scheduler_dino_lr_backbone[it]
            elif param_group['type'] == 'head':
                param_group["lr"] = scheduler_dino_lr_head[it]
            else:
                raise NotImplementedError
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule_dino[it]

        # move images to gpu
        images = [im.to(configs.device, non_blocking=True) for im in images]
        # teacher and student forward passes + compute dino loss
        # only the 2 global views pass through the teacher
        output_dino_head_teacher, features_teacher = teacher(images[:2])
        output_dino_head_student, features_student = student(images)

        loss_dino = dino_loss(output_dino_head_student, output_dino_head_teacher, epoch)
        if hasattr(configs.configs_dino, 'coef_loss_ce') and coef_loss_ce > 0:
            loss_ce = F.cross_entropy(features_student[:configs.batch_size_base], labels)
            loss = coef_loss_dino * loss_dino + coef_loss_ce * loss_ce
        else:
            loss = loss_dino

        if not math.isfinite(loss.item()):
            configs.logger.info("loss is %.4f, stopping training", loss.item())
            sys.exit(1)

        # Student update
        optimizer_dino.zero_grad()
        loss.backward()
        if configs.configs_dino.clip_grad:
            _ = dino_utils.clip_gradients(student, configs.configs_dino.clip_grad)
        dino_utils.cancel_gradients_last_layer(epoch, student, configs.configs_dino.num_epochs_freeze_last_layer)
        optimizer_dino.step()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule_dino[it]  # momentum parameter
            for param_q, param_k in zip(student.parameters(), teacher.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # Logging
        metric_logger.update(loss=loss.item())
        if hasattr(configs.configs_dino, 'coef_loss_ce') and coef_loss_ce > 0:
            metric_logger.update(loss_dino=loss_dino.item())
            metric_logger.update(loss_ce=loss_ce.item())
        metric_logger.update(lr=optimizer_dino.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer_dino.param_groups[0]["weight_decay"])
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def self_supervised_learning(configs: Configurations):
    # We create a root log directory. Next, we create a specific directory for this experiments. Therefore, each time we run
    #  the same experiment, we will get a new log file in its directory, and all log directories for different experiments
    # will be stored in one root directory.

    configs.model.train()

    _, loader_train_dino = get_dataloader_train(configs=configs, dino=True)

    _, loader_train_normal, _, loader_test_normal = get_datasets_and_dataloaders(configs=configs, dino=False)
    
    student = configs.get_the_model()
    teacher = configs.get_the_teacher()
    # Teacher and student start with the same weights
    # There is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    configs.logger.info("Student and Teacher are built.")

    num_epochs = configs.configs_dino.num_epochs
    
    # ============ preparing loss ... ============
    dino_loss = dino_utils.DINOLoss(
        configs.configs_dino.out_dim,
        # Total number of crops = 2 global crops + 8 local_crops_number
        configs.configs_dino.local_crops_number + 2,
        configs.configs_dino.warmup_teacher_temp,
        configs.configs_dino.teacher_temp,
        configs.configs_dino.warmup_teacher_temp_epochs,
        num_epochs,
    ).to(configs.device)

    # ============ preparing optimizer ... ============
    params_dino_student = dino_utils.get_params_groups(student)
    params_all = params_dino_student

    configs.optimizer_dino = torch.optim.AdamW(params_all)
    # ============ init schedulers ... ============

    # Their "base_value"s for the backbone and head schedulers are different
    configs.scheduler_dino_lr_backbone = dino_utils.cosine_scheduler(
        configs.configs_dino.lr_backbone,    # args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        configs.configs_dino.min_lr,
        num_epochs,
        len(loader_train_dino),
        warmup_epochs=configs.configs_dino.warmup_epochs_backbone,
    )
    
    configs.scheduler_dino_lr_head = dino_utils.cosine_scheduler(
        configs.configs_dino.lr_head,    # args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        configs.configs_dino.min_lr,
        num_epochs,
        len(loader_train_dino),
        warmup_epochs=configs.configs_dino.warmup_epochs_head,
    )

    configs.wd_schedule_dino = dino_utils.cosine_scheduler(
        configs.configs_dino.weight_decay,
        configs.configs_dino.weight_decay_end,
        num_epochs,
        len(loader_train_dino)
    )

    # momentum parameter is increased to 1. during training with a cosine schedule
    configs.momentum_schedule_dino = dino_utils.cosine_scheduler(configs.configs_dino.momentum_teacher,
                                                                 1,
                                                                 num_epochs,
                                                                 len(loader_train_dino)
                                                                 )
    configs.logger.info("Optimizer and schedulers are ready.")
    
    moving_average_capacity = configs.configs_logger.moving_average_capacity

    meters = MovingAverageDict(moving_average_capacity)
    sw = Stopwatch(['epoch', 'display', 'saved', 'total'])

    configs.logger.info("The base task training task has been started!")

    def display_losses(log_stats: dict):
        for k, v in log_stats.items():
            configs.logger.info("%s = %s", k, str(v))

    best_model = BestResult()
    configs.logger.info('-' * 50)
    sw.reset('total')

    def countdown(configs_dino, best_model, epoch):
        return configs_dino.num_epochs_no_progress_detection - (epoch - best_model.epoch)
    
    for epoch in range(configs.epoch, num_epochs):
        epochs_from_one = epoch + 1     # For displaying and saving
        sw.reset('epoch')
        configs.epoch = epoch
        train_stats = train_one_epoch(configs, student, teacher, dino_loss, loader_train_dino)
        
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}

        if epochs_from_one == num_epochs or epochs_from_one % configs.configs_save.save_freq_epoch == 0 or sw.elapsed_time('saved') >= configs.configs_save.time_interval_to_save * 60:
            configs.save(epochs_from_one, 0, snapshot=epochs_from_one % configs.configs_save.save_freq_epoch == 0)
            sw.reset('saved')

        display_losses(log_stats)

        if epoch == 0 or epoch == num_epochs - 1 or epoch % configs.configs_dino.fine_tuning_and_evaluation_freq == 0 or countdown(configs.configs_dino, best_model, epochs_from_one) < 10:
            acc = fine_tuning(configs=configs, loader_train_normal=loader_train_normal, loader_test_normal=loader_test_normal, evaluation_stochastic=False)     # evaluation_stochastic=False because we use a linear classifier head.
            if best_model.update(acc, epochs_from_one):
                configs.logger.info("This is the new best model!")
                configs.save(best_model.epoch, 0, is_this_the_best_model=True)
        
        configs.logger.info(best_model.to_string())

        student.train()
        teacher.train()

        configs.logger.info(f"This epoch took {sw.elapsed_time_in_hours_minutes('epoch')}!")
        configs.logger.info(f"The training took {sw.elapsed_time_in_hours_minutes('total')} from the beginning!")
        print_estimated_remaining_time(sw.total, num_epochs, epoch, display_func=configs.logger.info)

        meters.reset_all()

        # If not enough progress is observed, we stop training!
        if epochs_from_one - best_model.epoch > configs.configs_dino.num_epochs_no_progress_detection:
            configs.logger.info("The training is stopped due to the lack of progress!")
            break
        else:
            configs.logger.info(f"Number of epochs with no-progress = {epochs_from_one - best_model.epoch}/{configs.configs_dino.num_epochs_no_progress_detection}")
        configs.logger.info('-' * 50)
        # end of epoch
    configs.save(epoch=-1, iteration=-1)  # -1 stands for the latest
    configs.logger.info('The latest model is saved.')
    configs.logger.info(best_model.to_string())

    configs.logger.info('Training on the base task took %s.', sw.convert_to_hours_minutes(sw.elapsed_time('total')))


def supervised_learning(configs: Configurations):
    dataset_train, loader_train, dataset_test, loader_test = get_datasets_and_dataloaders(configs=configs, dino=False, task_id=0)

    model = configs.get_the_model()
    head = configs.get_the_classifier_head()
    
    for p in model.parameters():
        p.requires_grad = True
        
    freeze_the_first_layers(configs, model)

    model.train()
    head.train()

    configs_optimizer = configs.configs_train.optimizer
    params = dino_utils.get_params_groups(model, lr=configs_optimizer.lr_backbone) + dino_utils.get_params_groups(head, lr=configs_optimizer.lr)

    optimizer = prepare_optimizer(configs_optimizer, params, -1, configs.logger)
    
    scheduler_configs = configs.configs_train.scheduler

    scheduler = prepare_scheduler(scheduler_configs, optimizer)
    
    moving_average_capacity = configs.configs_logger.moving_average_capacity

    ma = MovingAverageDict(moving_average_capacity)
    sw = Stopwatch(['total', 'epoch', 'display'])

    configs.logger.info('-' * 50)
    num_epochs = configs.configs_train.num_epochs
    best_model = BestResult()
    sw.reset('total')

    it1 = range(configs.epoch, num_epochs)
    if configs.tqdm_enabled:
        it1 = tqdm(it1, position=0, leave=False, desc='Epoch', dynamic_ncols=True)
        
    evaluation_stochastic = configs.configs_train.evaluation_stochastic
    
    for epoch in it1:
        epochs_from_one = epoch + 1     # For displaying
        sw.reset('epoch')

        it2 = loader_train
        if configs.tqdm_enabled:
            it2 = tqdm(loader_train, position=1, leave=False, desc='Batch #', dynamic_ncols=True)

        for data in it2:
            images = data[0].to(configs.device)
            labels = data[1].to(configs.device)
            features = model(images)
            logits = head(features)

            logits = logits[:, : configs.configs_dataset.num_base_classes]

            loss = F.cross_entropy(logits, labels)
            ma.update(loss=loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # We only report the accuracy according to the display time interval only for the first epoch.
            if hasattr(configs.configs_logger, 'display_interval') and sw.display >= configs.configs_logger.display_interval * 60.0 and epoch == 0:
                sw.reset('display')
                configs.logger.info(f"MA loss: {ma['loss'].calculate()}")
                acc = test(configs, model, head, loader_test, 0, stochastic=evaluation_stochastic)

        # Save a snapshot
        if epochs_from_one % configs.configs_save.save_freq_epoch == 0 or epochs_from_one == num_epochs or sw.elapsed_time('saved') >= configs.configs_save.time_interval_to_save * 60:
            configs.save(epochs_from_one, 0, snapshot=True)
            sw.reset('saved')
        
        configs.logger.info("Epoch = %d" % epochs_from_one)
        msg = f"Mean average loss for the last {moving_average_capacity} iterations: {ma['loss'].calculate()}"
        configs.logger.info(msg)
        acc = test(configs, model, head, loader_test, 0, stochastic=evaluation_stochastic)
        if scheduler_configs.mode == 'max':
            scheduler.step(acc)
        else:
            scheduler.step(ma['loss'].calculate())
        if best_model.update(acc, epochs_from_one):
            configs.logger.info("This is the new best model!")
            configs.save(best_model.epoch, 0, is_this_the_best_model=True)
        configs.logger.info(best_model.to_string())
        configs.logger.info(f"This epoch took {sw.elapsed_time_in_hours_minutes('epoch')}!")
        configs.logger.info(f"The training took {sw.elapsed_time_in_hours_minutes('total')} from the beginning!")
        print_estimated_remaining_time(sw.total, num_epochs, epoch, display_func=configs.logger.info)

        if epochs_from_one - best_model.epoch > configs.configs_train.num_epochs_no_progress_detection:
            configs.logger.info("The training is stopped due to the lack of progress!")
            break
        else:
            configs.logger.info(f"Number of epochs with no-progress = {epochs_from_one - best_model.epoch}/{configs.configs_train.num_epochs_no_progress_detection}")

        configs.logger.info('-' * 50)
            
    configs.logger.info(best_model.to_string())
    configs.save(configs.configs_train.num_epochs, 0)
    configs.logger.info("Supervised learning phase is finished.")


def supervised_learning_with_prefixes(configs: Configurations):
    dataset_train, loader_train, dataset_test, loader_test = get_datasets_and_dataloaders(configs=configs, dino=False, task_id=0)

    model = Supervised_learning_with_prefixes(configs)
    head = configs.get_the_classifier_head()
    
    configs.model.zero_grad()

    model.train()
    head.train()

    configs_optimizer = configs.configs_train.optimizer
    params = list(model.parameters()) + list(head.parameters())

    optimizer = prepare_optimizer(configs_optimizer, params, -1, configs.logger)
    
    scheduler_configs = configs.configs_train.scheduler

    scheduler = prepare_scheduler(scheduler_configs, optimizer)
    
    moving_average_capacity = configs.configs_logger.moving_average_capacity

    ma = MovingAverageDict(moving_average_capacity)
    sw = Stopwatch(['total', 'epoch', 'display'])

    configs.logger.info('-' * 50)
    num_epochs = configs.configs_train.num_epochs
    best_model = BestResult()
    sw.reset('total')
    
    it1 = range(num_epochs)
    if configs.tqdm_enabled:
        it1 = tqdm(it1, position=0, leave=False, desc='Epoch', dynamic_ncols=True)
        
    evaluation_stochastic = configs.configs_train.evaluation_stochastic
    
    for epoch in it1:
        epochs_from_one = epoch + 1     # For displaying
        sw.reset('epoch')
        
        verification_model = verification(configs, model.backbone.blocks[0].attn.key.weight.data, False, "The backbone is supposed to be frozen!")
        
        verification_prefixes = verification(configs, configs.prefixes_base_task, True, "Error: We can not train the prefixes!")

        it2 = loader_train
        if configs.tqdm_enabled:
            it2 = tqdm(loader_train, position=1, leave=False, desc='Batch #', dynamic_ncols=True)

        for data in it2:
            images = data[0].to(configs.device)
            labels = data[1].to(configs.device)
            
            features = model(images)
            logits = head(features)

            logits = logits[:, : configs.configs_dataset.num_base_classes]

            loss = F.cross_entropy(logits, labels)
            ma.update(loss=loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            model.reset_backbone()
            
            # We only report the accuracy according to the display time interval only for the first epoch.
            if hasattr(configs.configs_logger, 'display_interval') and sw.display >= configs.configs_logger.display_interval * 60.0 and epoch == 0:
                sw.reset('display')
                configs.logger.info(f"MA loss: {ma['loss'].calculate()}")
                acc = test(configs, model, head, loader_test, 0, stochastic=evaluation_stochastic)

        verification_model.verify(model.backbone.blocks[0].attn.key.weight.data)
        verification_prefixes.verify(configs.prefixes_base_task)
        
        # Save a snapshot
        if epochs_from_one % configs.configs_save.save_freq_epoch == 0 or epochs_from_one == num_epochs or sw.elapsed_time('saved') >= configs.configs_save.time_interval_to_save * 60:
            configs.save(epochs_from_one, 0, snapshot=True)
            sw.reset('saved')
        
        configs.logger.info("Epoch = %d" % epoch)
        msg = f"Mean average loss for the last {moving_average_capacity} iterations: {ma['loss'].calculate()}"
        configs.logger.info(msg)
        
        acc = test(configs, model, head, loader_test, 0, stochastic=evaluation_stochastic)
        
        if scheduler_configs.mode == 'max':
            scheduler.step(acc)
        else:
            scheduler.step(ma['loss'].calculate())
        if best_model.update(acc, epochs_from_one):
            configs.logger.info("This is the new best model!")
            configs.save(best_model.epoch, 0, is_this_the_best_model=True)
        configs.logger.info(best_model.to_string())
        configs.logger.info(f"This epoch took {sw.elapsed_time_in_hours_minutes('epoch')}!")
        configs.logger.info(f"The training took {sw.elapsed_time_in_hours_minutes('total')} from the beginning!")
        print_estimated_remaining_time(sw.total, num_epochs, epoch, display_func=configs.logger.info)

        if epochs_from_one - best_model.epoch > configs.configs_train.num_epochs_no_progress_detection:
            configs.logger.info("The training is stopped due to the lack of progress!")
            break
        else:
            configs.logger.info(f"Number of epochs with no-progress = {epochs_from_one - best_model.epoch}/{configs.configs_train.num_epochs_no_progress_detection}")

        configs.logger.info('-' * 50)
            
    configs.logger.info(best_model.to_string())
    configs.save(configs.configs_train.num_epochs, 0)
    configs.logger.info("Supervised learning phase is finished.")


def fine_tuning(configs: Configurations, loader_train_normal, loader_test_normal, evaluation_stochastic):
    if loader_train_normal is None or loader_test_normal is None:
        _, loader_train_normal, _, loader_test_normal = get_datasets_and_dataloaders(configs, dino=False, task_id=0)

    if configs.phase == "fine_tuning":  # We automatially switch to the teacher model when we load the model from a file.
        model = configs.get_the_model()
    else:   # When we want to find the accuracy during the self-supervised learning phase.
        model = configs.get_the_teacher()

    configs.logger.info("Fine-tuning ...")
    
    if configs.phase == 'self_supervised_learning':
        freeze_backbone = configs.configs_dino.freeze_backbone_for_fine_tuning
    else:
        freeze_backbone = configs.configs_FSCIL.freeze_backbone

    model_state_old = model.training

    # Obtaining the features and labels lists

    inp_dim = configs.configs_model.embed_dim

    head = torch.nn.Linear(inp_dim, configs.configs_dataset.num_base_classes, device=configs.device)
    head.train()
    
    params = [{'params': head.parameters()}]

    if freeze_backbone:     # We compute the features one time, but train the classifier for many epochs
        model.eval()
        
        with torch.no_grad():
            features_list = []
            labels_list = []

            for data in loader_train_normal:
                images = data[0].to(configs.device)
                labels = data[1].to(configs.device)

                features = model(images)
                if configs.dino:     # Test during the training with DINO
                    assert len(features) == 2
                    features = features[1]
                features_list.append(features)
                labels_list.append(labels)

            features_all = torch.cat(features_list, dim=0)
            lebels_all = torch.cat(labels_list, dim=0)

            configs.logger.info("We obtianed the features from the frozen model.")

            ds_temp = TensorDataset(features_all, lebels_all)

            dataloader_temp = DataLoader(ds_temp,
                                        batch_size=configs.batch_size_fine_tuning,
                                        shuffle=True,
                                        drop_last=False)
    else:
        dataloader_temp = loader_train_normal
        model.train()
        params += dino_utils.get_params_groups(model)
    
    optimizer = torch.optim.Adam(params,
                                 lr=configs.configs_train.lr,
                                 betas=[configs.configs_train.momentum, configs.configs_train.momentum2],
                                 weight_decay=configs.configs_train.weight_decay)    # dampening=0.9, weight_decay=0.0

    moving_average_capacity = configs.configs_logger.moving_average_capacity
    
    ma = MovingAverageDict(moving_average_capacity)

    display_freq = configs.configs_train.display_freq

    it = range(configs.configs_train.num_epochs)
    if configs.tqdm_enabled:
        it = tqdm(it, dynamic_ncols=True)
        
    for epoch in it:
        for data in dataloader_temp:
            if freeze_backbone:
                features = data[0].to(configs.device)
                labels = data[1].to(configs.device)
            else:
                images = data[0].to(configs.device)
                labels = data[1].to(configs.device)
                features = model(images)
                if configs.dino:     # Test during the training with DINO
                    assert len(features) == 2
                    features = features[1]
            
            logits = head(features)
            loss = F.cross_entropy(logits, labels)
            ma.update(loss=loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % display_freq == 0 and not configs.dino:
            configs.logger.info(f"Mean average loss for the last {moving_average_capacity} iterations: {ma['loss'].calculate()}")
            if not freeze_backbone:
                configs.logger.info("Evaluation ...")
                test(configs, model, head, loader_test_normal, 0, stochastic=evaluation_stochastic)
            
    configs.logger.info("Final evaluation ...")
    res = test(configs, model, head, loader_test_normal, 0, stochastic=evaluation_stochastic)
    model.train(model_state_old)
    configs.collect_garbage()
    return res


def incremental_learning(configs: Configurations):
    dataset_train, loader_train, dataset_test, loader_test = get_datasets_and_dataloaders(configs, dino=False, task_id=0)
    task_id = 0     # The base learning task

    model = configs.get_the_model()

    freeze_the_first_layers(configs, model)

    acc_averager = AveragerDict()
    acc_averager_oracle = AveragerDict()

    trainer = FSCIL(configs)

    task_id_start = configs.configs_FSCIL.start_from_task

    sw = Stopwatch(['total'])

    for task_id in range(task_id_start, configs.configs_dataset.num_tasks):
        configs.set_task_id(task_id)
        configs.set_seed(configs.seed + task_id * 100)

        msg = f"The incremental learning phase for task {task_id} is started ..."
        configs.logger.info(msg)

        stats: DotMap = trainer.learn_this_task()
        if task_id == 0:
            acc = stats.accuracy.oracle
        else:
            acc = stats.accuracy.predictions
        acc_averager.add(task_id, acc)
        acc_averager_oracle.add(task_id, stats.accuracy.oracle)
        
        msg = f"The incremental learning phase for task {task_id} is finished!"
        configs.logger.info(msg)

        print_estimated_remaining_time(total_time=sw.total,
                                       total_tasks=configs.configs_dataset.num_tasks - task_id_start + 1,
                                       num_finished_tasks=task_id - task_id_start + 1,
                                       display_func=configs.logger.info)

    configs.logger.info('Final accuracies after each incremental task:')

    for i in range(configs.configs_dataset.num_tasks):
        data = acc_averager[i].data
        m = np.mean(data)
        # std = np.std(data)
        # ci = 1.96 * (std / np.sqrt(len(data)))
        message = f'Task {i}: {m * 100:.2f}'    # ± {ci * 100:.2f}
        configs.logger.info(message)

    configs.logger.info('The incremental learning phase is finished!')
    configs.logger.info(f"The whole process took {sw.elapsed_time_in_hours_minutes('total')}")


def test(configs: Configurations, model, head, loader_test, task_id: int, stochastic: bool, dataset_name: str = 'test', display=True):
    model = model.to(configs.device)
    head = head.to(configs.device)
    old_training_modes = DotMap()
    old_training_modes.model = model.training
    old_training_modes.head = head.training
    model.train(mode=False)
    head.train(mode=False)

    sum = 0.0
    count = 0.0
    rng = configs.get_learned_classes_and_current_task_range()

    with torch.no_grad():
        for data in loader_test:
            images = data[0].to(configs.device)
            labels = data[1].to(configs.device)
            
            features = model(images)
            if configs.dino:     # Test during the training with DINO
                features = features[1]

            if isinstance(head, StochasticClassifier) or isinstance(head, MLP_V):
                logits = head(features, stochastic)
            else:
                logits = head(features)
            
            predicted_labels = logits[:, rng[0]:rng[1]].argmax(dim=-1)

            sum += (predicted_labels == labels).sum()
            count += len(labels)

    acc = sum / count
    if display:
        log_str = f"-> {dataset_name} acc. of task {task_id}: {acc * 100:.2f}%"
        configs.logger.info(log_str)
    model.train(mode=old_training_modes.model)
    head.train(mode=old_training_modes.head)
    return acc


phase_str_to_func = {'self_supervised_learning': self_supervised_learning,
                     'incremental_learning': incremental_learning,
                     'fine_tuning': fine_tuning,
                     'supervised_learning': supervised_learning,
                     'supervised_learning_with_prefixes': supervised_learning_with_prefixes
                     }


class stdout_customized:
    def __init__(self, configs: Configurations):
        self.stdout = sys.stdout
        self.dir = os.path.join(".Logs", "temp")
        os.makedirs(self.dir, exist_ok=True)
        self.log_file_stdout = open(os.path.join(self.dir, f"stdout,{configs.time_str}.log"), "w")

    def write(self, text):
        self.stdout.write(text)
        self.log_file_stdout.write(text)

    def close(self):
        self.stdout.close()
        self.log_file_stdout.close()


def main():
    torch.multiprocessing.set_sharing_strategy('file_system')
    args = parse_arguments()

    configs = Configurations(args)
    configs.collect_garbage()
    configs.set_seed()
    
    stdout_obj = stdout_customized(configs)
    sys.stdout = stdout_obj

    try:
        phase_str_to_func[configs.phase](configs)
    except Exception as e:
        configs.logger.error(traceback.format_exc())
            
    stdout_obj.close()
    return 0


if __name__ == '__main__':
    main()

# References:

# @inproceedings{kalla2022s3c,
#   title={S3C: Self-supervised stochastic classifiers for few-shot class-incremental learning},
#   author={Kalla, Jayateja and Biswas, Soma},
#   booktitle={European Conference on Computer Vision},
#   pages={432--448},
#   year={2022},
#   organization={Springer}
# }

# We used some codes from the implementation of the following references and tried to conform to the conditions in their experiments:
# @misc{rw2019timm,
#   author = {Ross Wightman},
#   title = {PyTorch Image Models},
#   year = {2019},
#   publisher = {GitHub},
#   journal = {GitHub repository},
#   doi = {10.5281/zenodo.4414861},
#   howpublished = {\url{https://github.com/rwightman/pytorch-image-models}}
# }

# @article{Caron2021EmergingPI,
#   title={Emerging Properties in Self-Supervised Vision Transformers},
#   author={Mathilde Caron and Hugo Touvron and Ishan Misra and Herv'e J'egou and Julien Mairal and Piotr Bojanowski and Armand Joulin},
#   journal={2021 IEEE/CVF International Conference on Computer Vision (ICCV)},
#   year={2021},
#   pages={9630-9640}
# }

# We may also have used some codes from the implementations of the following papers!

# @article{Xu2021CDTransCT,
#   title={CDTrans: Cross-domain Transformer for Unsupervised Domain Adaptation},
#   author={Tongkun Xu and Weihua Chen and Pichao Wang and Fan Wang and Hao Li and Rong Jin},
#   journal={ArXiv},
#   year={2021},
#   volume={abs/2109.06165}
# }

# @article{Yang2021TVTTV,
#   title={TVT: Transferable Vision Transformer for Unsupervised Domain Adaptation},
#   author={Jinyu Yang and Jingjing Liu and Ning Xu and Junzhou Huang},
#   journal={2023 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
#   year={2021},
#   pages={520-530}
# }
