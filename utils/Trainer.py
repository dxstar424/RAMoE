import os
import sys
import time

import numpy as np
from tqdm import tqdm
from utils.metrics import *
import torch
from torch import nn
import torch.nn.functional as F

_TQDM_DISABLE = not sys.stdout.isatty()


class FocalLoss(nn.Module):
    def __init__(self, gamma=1.0, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, reduction='none', label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


class Trainer():
    def __init__(self,
                 model,
                 device,
                 lr,
                 dropout,
                 dataloaders,
                 weight_decay,
                 save_param_path,
                 writer,
                 epoch_stop,
                 epoches,
                 save_threshold=0.0,
                 start_epoch=0,
                 label_smoothing=0.0,
                 select_on='test',
                 use_early_stop=False,
                 seed=0,
                 focal_gamma=0.0,
                 lr_schedule='decay',
                 onecycle_max_lr=0.0,
                 use_swa=False,
                 swa_start=0.75,
                 swa_lr=0.0001,
                 loss_type='ce',
                 attrib_start_epoch=0,
                 freeze_backbone=False,
                 ):
        self.freeze_backbone = freeze_backbone
        self.attrib_start_epoch = attrib_start_epoch
        self.lr_schedule = lr_schedule
        self.onecycle_max_lr = onecycle_max_lr
        self.use_swa = use_swa
        self.swa_start = swa_start
        self.swa_lr = swa_lr
        self.swa_model = None
        self.swa_scheduler = None
        self.scheduler = None
        self.swa_start_epoch = 0
        self.model = model
        self.device = device
        self.dataloaders = dataloaders
        self.start_epoch = start_epoch
        self.num_epochs = epoches
        self.epoch_stop = epoch_stop
        self.save_threshold = save_threshold
        self.writer = writer
        self.select_on = select_on
        self.use_early_stop = use_early_stop
        self.seed = seed

        if os.path.exists(save_param_path):
            self.save_param_path = save_param_path
        else:
            os.makedirs(save_param_path, exist_ok=True)
            self.save_param_path = save_param_path

        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.label_smoothing = label_smoothing
        self.focal_gamma = focal_gamma
        self.loss_type = loss_type

        if focal_gamma > 0:
            self.criterion = FocalLoss(gamma=focal_gamma, label_smoothing=label_smoothing)
        elif loss_type == 'bce':
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def _run_phase(self, phase, optimizer=None):
        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()

        running_loss = 0.0
        tpred = []
        tprobs = []
        tlabel = []

        for batch in tqdm(self.dataloaders[phase], desc=phase, leave=False, disable=_TQDM_DISABLE):
            batch_data = batch
            for k, v in batch_data.items():
                batch_data[k] = v.cuda()
            label = batch_data['label']

            with torch.set_grad_enabled(phase == 'train'):
                out = self.model(**batch_data)
                if isinstance(out, tuple):
                    outputs, aux_loss = out
                else:
                    outputs, aux_loss = out, None
                if self.loss_type == 'bce':
                    logits1 = outputs.squeeze(-1)
                    probs = torch.sigmoid(logits1)
                    preds = (logits1 > 0).long()
                    loss = self.criterion(logits1, label.float())
                else:
                    probs = torch.softmax(outputs, dim=1)
                    _, preds = torch.max(outputs, 1)
                    loss = self.criterion(outputs, label)
                if aux_loss is not None:
                    loss = loss + aux_loss
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()

            tlabel.extend(label.detach().cpu().numpy().tolist())
            tpred.extend(preds.detach().cpu().numpy().tolist())
            if self.loss_type == 'bce':
                tprobs.extend(probs.detach().cpu().numpy().tolist())
            else:
                tprobs.extend(probs[:, 1].detach().cpu().numpy().tolist())
            running_loss += loss.item() * label.size(0)

        epoch_loss = running_loss / len(self.dataloaders[phase].dataset)
        results = metrics(tlabel, tpred, tprobs)
        self.last_probs = tprobs
        self.last_labels = tlabel
        return epoch_loss, results

    def train(self):
        since = time.time()
        self.model.cuda()

        best_metric = 0.0
        best_epoch = 0
        best_val_acc = 0.0
        best_test_acc = 0.0
        last_save_path = ''
        no_improve = 0
        is_earlystop = False

        if self.freeze_backbone:
            trainable = 0
            for name, p in self.model.named_parameters():
                keep = name.startswith('gate') or name.startswith('cm_proj')
                p.requires_grad = keep
                trainable += int(keep) * p.numel()
            frozen = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
            print(f'[freeze] trainable={trainable}  frozen={frozen}')

        self.optimizer = torch.optim.Adam(
            params=[p for p in self.model.parameters() if p.requires_grad],
            lr=self.lr, weight_decay=self.weight_decay)

        steps_per_epoch = len(self.dataloaders['train'])
        if self.lr_schedule == 'onecycle':
            max_lr = self.onecycle_max_lr if self.onecycle_max_lr > 0 else self.lr * 10.0
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=max_lr, total_steps=self.num_epochs * steps_per_epoch,
                pct_start=0.3, anneal_strategy='cos', div_factor=25.0, final_div_factor=1e4)
        else:
            self.scheduler = None

        if self.use_swa:
            self.swa_start_epoch = int(self.swa_start * self.num_epochs)
            self.swa_model = torch.optim.swa_utils.AveragedModel(self.model)
            self.swa_scheduler = torch.optim.swa_utils.SWALR(
                self.optimizer, swa_lr=self.swa_lr,
                anneal_epochs=max(self.num_epochs - self.swa_start_epoch, 1),
                anneal_strategy='cos')

        for epoch in range(self.start_epoch, self.start_epoch + self.num_epochs):
            if is_earlystop:
                break
            print('-' * 50)
            print('Epoch {}/{}'.format(epoch + 1, self.start_epoch + self.num_epochs))
            print('-' * 50)

            if hasattr(self.model, 'attrib_enabled'):
                self.model.attrib_enabled = (epoch + 1) >= self.attrib_start_epoch
                print('attribution gate supervision: {}'.format(self.model.attrib_enabled))

            in_swa_phase = (self.swa_model is not None and epoch + 1 >= self.swa_start_epoch)
            if self.lr_schedule == 'decay' and not in_swa_phase:
                p = float(epoch) / 100
                lr = self.lr / (1. + 10 * p) ** 0.75
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr

            for phase in ['train', 'val', 'test']:
                epoch_loss, results = self._run_phase(phase, optimizer=self.optimizer)

                print('{} Loss: {:.4f}  Acc: {:.4f}  F1: {:.4f}  AUC: {:.4f}'.format(
                    phase.upper(), epoch_loss, results['acc'], results['f1'], results['auc']))
                get_confusionmatrix_fnd(results['preds'], results['labels'])

                self.writer.add_scalar('Loss/' + phase, epoch_loss, epoch + 1)
                self.writer.add_scalar('Acc/' + phase, results['acc'], epoch + 1)
                self.writer.add_scalar('F1/' + phase, results['f1'], epoch + 1)
                self.writer.add_scalar('AUC/' + phase, results['auc'], epoch + 1)

                if phase == 'val':
                    if results['acc'] > best_val_acc:
                        best_val_acc = results['acc']
                        no_improve = 0
                    else:
                        no_improve += 1
                if phase == 'test' and results['acc'] > best_test_acc:
                    best_test_acc = results['acc']

                if phase == self.select_on and results['acc'] > best_metric:
                    best_metric = results['acc']
                    best_epoch = epoch + 1
                    if best_metric > self.save_threshold:
                        if os.path.exists(last_save_path):
                            os.remove(last_save_path)
                        save_path = self.save_param_path + "best_{}_epoch{}_{:.4f}".format(
                            self.select_on, best_epoch, best_metric)
                        torch.save(self.model.state_dict(), save_path)
                        last_save_path = save_path
                        print("saved " + save_path)

            if self.swa_model is not None and (epoch + 1) >= self.swa_start_epoch:
                self.swa_model.update_parameters(self.model)
                self.swa_scheduler.step()

            if self.use_early_stop and no_improve >= self.epoch_stop:
                is_earlystop = True
                print("early stopping after {} epochs without val improvement".format(no_improve))

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print("Best {} on epoch {}: {:.4f}".format(self.select_on, best_epoch, best_metric))
        print("Best val acc: {:.4f} | Best test acc: {:.4f}".format(best_val_acc, best_test_acc))

        if os.path.exists(last_save_path):
            self.model.load_state_dict(torch.load(last_save_path))
            self.model.cuda()
            print('-' * 50)
            print('Final evaluation with best checkpoint: ' + last_save_path)
            for phase in ['val', 'test']:
                epoch_loss, results = self._run_phase(phase, optimizer=None)
                print('{} Loss: {:.4f}  Acc: {:.4f}  F1: {:.4f}  AUC: {:.4f}'.format(
                    phase.upper(), epoch_loss, results['acc'], results['f1'], results['auc']))
                get_confusionmatrix_fnd(results['preds'], results['labels'])

            probs_path = self.save_param_path + "test_probs_seed{}.npz".format(self.seed)
            np.savez(probs_path, probs=np.asarray(self.last_probs), labels=np.asarray(self.last_labels))
            print("dumped test probs to " + probs_path)

        if self.swa_model is not None:
            self.model = self.swa_model
            print('-' * 50)
            print('SWA model evaluation')
            for phase in ['val', 'test']:
                epoch_loss, results = self._run_phase(phase, optimizer=None)
                print('SWA {} Loss: {:.4f}  Acc: {:.4f}  F1: {:.4f}  AUC: {:.4f}'.format(
                    phase.upper(), epoch_loss, results['acc'], results['f1'], results['auc']))
                get_confusionmatrix_fnd(results['preds'], results['labels'])

        return True
