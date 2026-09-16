import argparse
import os
import random
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import torch

from run import Run

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='fakesv', help='fakesv/fakett')
parser.add_argument('--epoches', type=int, default=None)
parser.add_argument('--batch_size', type=int, default=None)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--seed', type=int, default=2024)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--lr', type=float, default=None)
parser.add_argument('--dropout', type=float, default=None)
parser.add_argument('--weight_decay', type=float, default=None)
parser.add_argument('--label_smoothing', type=float, default=None)
parser.add_argument('--select_on', default='test', help='test/val')
parser.add_argument('--use_early_stop', type=int, default=0)
parser.add_argument('--out_dim', type=int, default=256)
parser.add_argument('--max_text_len', type=int, default=512)
parser.add_argument('--activation', default='relu', help='relu/silu/gelu')
parser.add_argument('--lr_schedule', default='decay', help='decay/onecycle')
parser.add_argument('--loss_type', default='ce', help='ce/bce')

parser.add_argument('--gate_hidden', type=int, default=64)
parser.add_argument('--gate_layers', type=int, default=2)
parser.add_argument('--gate_dropout', type=float, default=None)
parser.add_argument('--gate_cm_proj', type=int, default=0,
                    help='1 = append a small projection of the co-attention features to the gate input')
parser.add_argument('--gate_cm_dim', type=int, default=16)
parser.add_argument('--gate_weight', default='sqrt_inv', help='none/inv/sqrt_inv class weighting of L_attr')

parser.add_argument('--use_attrib', type=int, default=1, help='1 = counterfactual attribution labels for the gate')
parser.add_argument('--lambda_single', type=float, default=0.5)
parser.add_argument('--lambda_interaction', type=float, default=0.3)
parser.add_argument('--attrib_margin', type=float, default=0.5, help='absolute margin s* - s2 for declaring a single culprit')
parser.add_argument('--attrib_relative', type=float, default=0.0, help='if >0, use s* >= rho * s2 instead of the absolute margin')
parser.add_argument('--attrib_start_epoch', type=int, default=3, help='warm-up: first k epochs train without L_attr')

parser.add_argument('--offline_labels', default='', help='.npy of fixed attribution labels produced by gen_labels.py (stage 2)')
parser.add_argument('--init_from', default='', help='checkpoint used to initialise the model (stage 1 -> stage 2)')
parser.add_argument('--attrib_ref', default='gate', help='gate/uniform: dispatch used in the teacher attribution')
parser.add_argument('--attrib_mode', default='full', help='expertmean (in-dist batch-mean replacement) / selfmean / full (zero) / expert (zero slices)')
parser.add_argument('--label_source', default='', help='free-text tag stored in the run config')
parser.add_argument('--freeze_backbone', type=int, default=0,
                    help='1 = stage 2 trains only the gate (and cm_proj), keeping the '
                         'stage-1 detector fixed so attribution cannot disturb classification')

parser.add_argument('--path_param', default='check_points/')
parser.add_argument('--path_tensorboard', default='tb/')
args = parser.parse_args()

_DEFAULTS = dict(lr=3e-4, batch_size=32, dropout=0.2, weight_decay=1e-4,
                 label_smoothing=0.1, epoches=30)
for _k, _v in _DEFAULTS.items():
    if getattr(args, _k) is None:
        setattr(args, _k, _v)

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

print(args)

config = dict(
    dataset=args.dataset, epoches=args.epoches, batch_size=args.batch_size,
    num_workers=args.num_workers, seed=args.seed, device=args.gpu, lr=args.lr,
    dropout=args.dropout, weight_decay=args.weight_decay,
    label_smoothing=args.label_smoothing, select_on=args.select_on,
    use_early_stop=bool(args.use_early_stop), out_dim=args.out_dim,
    max_text_len=args.max_text_len, activation=args.activation,
    lr_schedule=args.lr_schedule, loss_type=args.loss_type,
    gate_hidden=args.gate_hidden, gate_layers=args.gate_layers,
    gate_dropout=args.gate_dropout, gate_cm_proj=bool(args.gate_cm_proj),
    gate_cm_dim=args.gate_cm_dim, gate_weight=args.gate_weight,
    use_attrib=bool(args.use_attrib), lambda_single=args.lambda_single,
    lambda_interaction=args.lambda_interaction, attrib_margin=args.attrib_margin,
    attrib_relative=args.attrib_relative, attrib_start_epoch=args.attrib_start_epoch,
    offline_labels=args.offline_labels, init_from=args.init_from,
    attrib_ref=args.attrib_ref, label_source=args.label_source,
    attrib_mode=args.attrib_mode, freeze_backbone=bool(args.freeze_backbone),
    path_param=args.path_param, path_tensorboard=args.path_tensorboard,
)

if __name__ == '__main__':
    Run(config=config).main()
