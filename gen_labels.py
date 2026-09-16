import os, sys, json, argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.ramoe import RAMoE
from utils.dataloader import MFSVFNDDataset

DEV = "cuda" if torch.cuda.is_available() else "cpu"
ROLE = ["text", "visual", "audio", "joint", "authentic"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--ckpt', required=True, help='frozen teacher checkpoint')
    ap.add_argument('--out', required=True, help='output .npy with the labels')
    ap.add_argument('--calib', default='relative', choices=['relative', 'absolute', 'off'])
    ap.add_argument('--target_joint', type=float, default=0.35)
    ap.add_argument('--attrib_relative', type=float, default=1.5)
    ap.add_argument('--attrib_margin', type=float, default=0.5)
    ap.add_argument('--attrib_ref', default='gate', choices=['gate', 'uniform'])
    ap.add_argument('--attrib_mode', default='full', choices=['full', 'expert', 'selfmean', 'expertmean'])
    ap.add_argument('--criterion', default='signed', choices=['signed', 'magnitude'],
                    help='signed: culprit = argmax s (drop in the fake logit); '
                         'magnitude: culprit = argmax |s| (influence, robust to OOD sign flips)')
    ap.add_argument('--gate_hidden', type=int, default=64)
    ap.add_argument('--gate_layers', type=int, default=2)
    ap.add_argument('--gate_cm_proj', type=int, default=0)
    ap.add_argument('--batch_size', type=int, default=32)
    a = ap.parse_args()

    ds = MFSVFNDDataset('vid_time3_train.txt', a.dataset, max_text_len=512)
    if a.dataset == 'fakesv':
        from utils.dataloader import fakesv_collate_fn as collate
    else:
        from utils.dataloader import fakett_collate_fn as collate
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=False, drop_last=False,
                        num_workers=0, collate_fn=collate)

    ck = torch.load(a.ckpt, map_location='cpu')
    tdim = ck['enc_text.proj.weight'].shape[1]
    model = RAMoE(dataset=a.dataset, text_dim=tdim, gate_hidden=a.gate_hidden,
                        gate_layers=a.gate_layers, gate_cm_proj=bool(a.gate_cm_proj))
    missing, unexpected = model.load_state_dict(ck, strict=False)
    model.attrib_ref = a.attrib_ref
    model.attrib_mode = a.attrib_mode
    model.eval().to(DEV)
    print(f'teacher: {os.path.basename(a.ckpt)}  (missing={len(missing)}, unexpected={len(unexpected)})')

    Y, S = [], []
    with torch.no_grad():
        for batch in loader:
            b = {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in batch.items()}
            out = model.diagnose(**b)
            Y.append(b['label'].cpu().numpy())
            S.append(out['s'].cpu().numpy())
    Y = np.concatenate(Y); S = np.concatenate(S)
    n = len(Y)
    print(f'train samples: {n};  real={int((Y==0).sum())}, fake={int((Y==1).sum())}')

    fake = Y == 1
    sf = S[fake]
    if a.criterion == 'magnitude':
        sf = np.abs(sf)
    order = np.argsort(-sf, axis=1)
    s1 = sf[np.arange(len(sf)), order[:, 0]]
    s2 = sf[np.arange(len(sf)), order[:, 1]]
    arg = order[:, 0]
    gap = s1 - s2
    ratio = s1 / np.maximum(s2, 1e-6)
    floor = 1e-3 if a.criterion == 'magnitude' else 0.0

    if a.calib == 'relative':
        rho = float(np.quantile(ratio, a.target_joint))
        single = (s1 > floor) & (ratio >= rho)
        chosen = dict(mode='relative', rho=rho)
    elif a.calib == 'absolute':
        margin = float(np.quantile(gap, a.target_joint))
        single = (s1 > floor) & (gap > margin)
        chosen = dict(mode='absolute', margin=margin)
    else:
        rho = a.attrib_relative
        single = (s1 > floor) & (ratio >= rho) if rho > 0 else (s1 > floor) & (gap > a.attrib_margin)
        chosen = dict(mode='off', rho=rho, margin=a.attrib_margin)

    labels = np.full(n, 4, dtype=np.int64)
    fake_idx = np.where(fake)[0]
    labels[fake_idx] = np.where(single, arg, 3)

    dist = {ROLE[i]: int((labels == i).sum()) for i in range(5)}
    fake_dist = {ROLE[i]: int(((labels == i) & fake).sum()) for i in range(5)}
    joint_frac = float(((labels == 3) & fake).sum() / max(int(fake.sum()), 1))
    stats = dict(dataset=a.dataset, teacher=os.path.basename(a.ckpt), n=n,
                 calib=a.calib, criterion=a.criterion, attrib_mode=a.attrib_mode,
                 target_joint=a.target_joint, chosen=chosen,
                 label_dist=dist, fake_label_dist=fake_dist,
                 joint_fraction_among_fake=joint_frac,
                 frac_s1_nonpositive=float((s1 <= 0).mean()),
                 mean_s=[float(S[:, 0].mean()), float(S[:, 1].mean()), float(S[:, 2].mean())])
    np.save(a.out, labels)
    with open(os.path.splitext(a.out)[0] + '_stats.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print('chosen threshold :', chosen)
    print('label dist       :', dist)
    print('fake label dist  :', fake_dist)
    print(f'joint among fake : {joint_frac*100:.1f}%   (target {a.target_joint*100:.0f}%)')
    print('mean s           :', [round(float(S[:, i].mean()), 3) for i in range(3)])
    print('saved ->', a.out)


if __name__ == '__main__':
    main()
