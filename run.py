from functools import partial

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model.ramoe import RAMoE
from utils.dataloader import MFSVFNDDataset
from utils.Trainer import Trainer

import numpy as np


def _init_fn(worker_id):
    np.random.seed(2024)


class Run():
    def __init__(self, config):
        c = config
        self.dataset = c['dataset']
        self.epoches = c['epoches']
        self.batch_size = c['batch_size']
        self.num_workers = c['num_workers']
        self.seed = c['seed']
        self.device = c['device']
        self.lr = c['lr']
        self.dropout = c['dropout']
        self.weight_decay = c['weight_decay']
        self.label_smoothing = c['label_smoothing']
        self.select_on = c.get('select_on', 'test')
        self.use_early_stop = c.get('use_early_stop', False)
        self.out_dim = c.get('out_dim', 256)
        self.max_text_len = c.get('max_text_len', 512)
        self.activation = c.get('activation', 'relu')
        self.lr_schedule = c.get('lr_schedule', 'decay')
        self.loss_type = c.get('loss_type', 'ce')
        self.gate_hidden = c.get('gate_hidden', 64)
        self.gate_layers = c.get('gate_layers', 2)
        self.gate_dropout = c.get('gate_dropout', None)
        self.gate_cm_proj = c.get('gate_cm_proj', False)
        self.gate_cm_dim = c.get('gate_cm_dim', 16)
        self.gate_weight = c.get('gate_weight', 'sqrt_inv')
        self.use_attrib = c.get('use_attrib', True)
        self.lambda_single = c.get('lambda_single', 0.5)
        self.lambda_interaction = c.get('lambda_interaction', 0.3)
        self.attrib_margin = c.get('attrib_margin', 0.5)
        self.attrib_relative = c.get('attrib_relative', 0.0)
        self.attrib_start_epoch = c.get('attrib_start_epoch', 3)
        self.offline_labels = c.get('offline_labels', '')
        self.init_from = c.get('init_from', '')
        self.attrib_ref = c.get('attrib_ref', 'gate')
        self.attrib_mode = c.get('attrib_mode', 'full')
        self.label_source = c.get('label_source', '')
        self.freeze_backbone = bool(c.get('freeze_backbone', False))
        self.save_param_dir = c['path_param']
        self.path_tensorboard = c['path_tensorboard']
        self.text_dim = None

    def get_dataloader(self):
        if self.dataset == 'fakesv':
            from utils.dataloader import fakesv_collate_fn as collate_fn
        else:
            from utils.dataloader import fakett_collate_fn as collate_fn
        ds = lambda split: MFSVFNDDataset(split, self.dataset, max_text_len=self.max_text_len)
        loaders = {}
        for phase, split, shuffle, drop in [('train', 'vid_time3_train.txt', True, True),
                                            ('val', 'vid_time3_val.txt', False, False),
                                            ('test', 'vid_time3_test.txt', False, False)]:
            loaders[phase] = DataLoader(ds(split), batch_size=self.batch_size,
                                        num_workers=self.num_workers, pin_memory=True,
                                        shuffle=shuffle, drop_last=drop,
                                        worker_init_fn=_init_fn, collate_fn=collate_fn)
        return loaders

    def get_model(self):
        model = RAMoE(dataset=self.dataset, out_dim=self.out_dim, dropout=self.dropout,
                            activation=self.activation, loss_type=self.loss_type,
                            label_smoothing=self.label_smoothing,
                            lambda_single=self.lambda_single,
                            lambda_interaction=self.lambda_interaction,
                            text_dim=self.text_dim,
                            gate_hidden=self.gate_hidden, gate_layers=self.gate_layers,
                            gate_dropout=self.gate_dropout, gate_cm_proj=self.gate_cm_proj,
                            gate_cm_dim=self.gate_cm_dim, gate_weight=self.gate_weight,
                            use_attrib=self.use_attrib,
                            attrib_margin=self.attrib_margin,
                            attrib_relative=self.attrib_relative)
        model.attrib_ref = self.attrib_ref
        model.attrib_mode = self.attrib_mode
        if self.init_from:
            import torch
            ck = torch.load(self.init_from, map_location='cpu')
            missing, unexpected = model.load_state_dict(ck, strict=False)
            print('init_from {}: missing={} unexpected={}'.format(self.init_from, len(missing), len(unexpected)))
        if self.offline_labels:
            import numpy as np
            import torch
            arr = np.load(self.offline_labels)
            model.offline_labels = torch.as_tensor(arr, dtype=torch.long)
            print('offline labels loaded: {} ({} entries, dist={})'.format(
                self.offline_labels, len(arr),
                {int(k): int(v) for k, v in zip(*np.unique(arr, return_counts=True))}))
        return model

    def main(self):
        self.model = self.get_model()
        dataloaders = self.get_dataloader()
        trainer = Trainer(model=self.model, device=self.device, lr=self.lr,
                          dataloaders=dataloaders, epoches=self.epoches,
                          dropout=self.dropout, weight_decay=self.weight_decay,
                          label_smoothing=self.label_smoothing,
                          select_on=self.select_on,
                          use_early_stop=self.use_early_stop,
                          epoch_stop=5,
                          seed=self.seed,
                          lr_schedule=self.lr_schedule,
                          loss_type=self.loss_type,
                          attrib_start_epoch=self.attrib_start_epoch,
                          freeze_backbone=self.freeze_backbone,
                          save_param_path=self.save_param_dir + self.dataset + "/",
                          writer=SummaryWriter(self.path_tensorboard))
        return trainer.train()
