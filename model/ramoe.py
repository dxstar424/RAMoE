import torch
import torch.nn as nn
import torch.nn.functional as F

_ACT = {'relu': nn.ReLU, 'silu': nn.SiLU, 'gelu': nn.GELU}
_ACT_FN = {'relu': F.relu, 'silu': F.silu, 'gelu': F.gelu}


_BLOCK_SRC = [
    [{'t'}, {'t', 'v'}, {'t', 'a'}],
    [{'v'}, {'t', 'v'}, {'v', 'a'}],
    [{'a'}, {'t', 'a'}, {'v', 'a'}],
    [{'t', 'v'}, {'t', 'a'}, {'v', 'a'}],
    [{'t'}, {'v'}, {'a'}],
]
ROLE_NAMES = ['text', 'visual', 'audio', 'joint', 'authentic']


class AttentionPool(nn.Module):
    def __init__(self, dim, n_queries=4):
        super().__init__()
        self.queries = nn.Parameter(torch.zeros(n_queries, dim))
        nn.init.normal_(self.queries, std=0.01)
        self.scale = dim ** 0.5

    def forward(self, x, mask=None):
        attn = torch.einsum('bld,kd->blk', x, self.queries) / self.scale
        if mask is not None:
            attn = attn.masked_fill((mask == 0).unsqueeze(-1), -1e9)
        attn = F.softmax(attn, dim=1)
        return torch.einsum('blk,bld->bkd', attn, x).mean(dim=1)


class ModalityEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1, nhead=4, n_queries=4,
                 num_layers=1, activation='relu'):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.act = _ACT_FN[activation]
        layer = nn.TransformerEncoderLayer(d_model=out_dim, nhead=nhead,
                                           dim_feedforward=out_dim * 2,
                                           dropout=dropout, batch_first=True)
        self.self_attn = nn.TransformerEncoder(layer, num_layers=num_layers,
                                               enable_nested_tensor=False)
        self.pool = AttentionPool(out_dim, n_queries=n_queries)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.dropout(self.act(self.proj(x)))
        pad_mask = (mask == 0) if mask is not None else None
        x = self.self_attn(x, src_key_padding_mask=pad_mask)
        return self.pool(x, mask), x


class CoAttention(nn.Module):
    def __init__(self, dim, nhead=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y, y_mask=None):
        key_pad = (y_mask == 0) if y_mask is not None else None
        out, _ = self.attn(x, y, y, key_padding_mask=key_pad)
        return self.norm(x + self.dropout(out))


class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout=0.1, activation='relu', n_layers=2):
        super().__init__()
        act = _ACT[activation]
        layers, d = [], in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), act(), nn.Dropout(dropout)]
            d = hidden
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class RAMoE(nn.Module):
    def __init__(self, dataset, out_dim=256, dropout=0.1, activation='relu',
                 lambda_single=0.5, lambda_interaction=0.3, label_smoothing=0.0,
                 loss_type='ce', text_dim=None,
                 # gate
                 gate_hidden=64, gate_layers=2, gate_dropout=None,
                 gate_cm_proj=False, gate_cm_dim=16,
                 # attribution supervision
                 use_attrib=True, attrib_margin=0.5, attrib_relative=0.0,
                 gate_weight='sqrt_inv', gate_ema=0.05, attrib_which='all',
                 attrib_mode='full'):
        super().__init__()
        if text_dim is None:
            text_dim = 1024 if dataset == 'fakesv' else 768
        self.dataset = dataset
        self.out_dim = out_dim
        self.num_experts = 5
        self.lambda_single = lambda_single
        self.lambda_interaction = lambda_interaction
        self.loss_type = loss_type
        self.use_attrib = use_attrib
        self.attrib_margin = attrib_margin
        self.attrib_relative = attrib_relative
        self.gate_weight = gate_weight
        self.gate_ema = gate_ema
        self.gate_cm_proj = gate_cm_proj
        self.attrib_enabled = False
        self.force_uniform = False
        self.offline_labels = None
        self.register_buffer('_offline_labels', None)
        self.attrib_ref = 'gate'
        self.attrib_mode = attrib_mode
        task_out = 2 if loss_type == 'ce' else 1
        self.task_out = task_out

        self.enc_text = ModalityEncoder(text_dim, out_dim, dropout, activation=activation)
        self.enc_visual = ModalityEncoder(1024, out_dim, dropout, activation=activation)
        self.enc_audio = ModalityEncoder(1024, out_dim, dropout, activation=activation)

        self.co_tv = CoAttention(out_dim, dropout=dropout)
        self.co_vt = CoAttention(out_dim, dropout=dropout)
        self.co_ta = CoAttention(out_dim, dropout=dropout)
        self.co_at = CoAttention(out_dim, dropout=dropout)
        self.co_va = CoAttention(out_dim, dropout=dropout)
        self.co_av = CoAttention(out_dim, dropout=dropout)

        self.cls_text = MLP(out_dim, 64, 2, dropout, activation, n_layers=1)
        self.cls_visual = MLP(out_dim, 64, 2, dropout, activation, n_layers=1)
        self.cls_audio = MLP(out_dim, 64, 2, dropout, activation, n_layers=1)

        gate_in = 3 + (gate_cm_dim if gate_cm_proj else 0)
        self.cm_proj = (nn.Sequential(nn.Linear(3 * out_dim, gate_cm_dim), _ACT[activation]())
                        if gate_cm_proj else None)
        self.gate = MLP(gate_in, gate_hidden, self.num_experts,
                        dropout if gate_dropout is None else gate_dropout,
                        activation, n_layers=gate_layers)

        expert_in = 3 * out_dim
        self.experts = nn.ModuleList([MLP(expert_in, 128, task_out, dropout, activation, n_layers=3)
                                      for _ in range(self.num_experts)])

        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.plain_ce = nn.CrossEntropyLoss()
        self.register_buffer('gate_counts', torch.ones(self.num_experts))

    @staticmethod
    def _cosine(a, b):
        return (F.normalize(a, dim=-1) * F.normalize(b, dim=-1)).sum(dim=-1)

    def _cross(self, co_a2b, co_b2a, xa, xb, xa_mask, xb_mask, pool_a, pool_b):
        o_ab = pool_a(co_a2b(xa, xb, xb_mask))
        o_ba = pool_b(co_b2a(xb, xa, xa_mask))
        return o_ab + o_ba

    def _blocks(self, h_t, h_v, h_a, cm_tv, cm_ta, cm_va):
        return [
            [h_t, cm_tv, cm_ta],
            [h_v, cm_tv, cm_va],
            [h_a, cm_ta, cm_va],
            [cm_tv, cm_ta, cm_va],
            [h_t, h_v, h_a],
        ]

    def _expert_logits(self, blocks):
        logits = []
        for i, blk in enumerate(blocks):
            logits.append(self.experts[i](torch.cat(blk, dim=-1)))
        return torch.stack(logits, dim=1)

    def _ablated_blocks(self, blocks, modality):
        out = []
        for i, blk in enumerate(blocks):
            new = []
            for j, b in enumerate(blk):
                if modality in _BLOCK_SRC[i][j]:
                    if self.attrib_mode == 'expertmean':
                        new.append(b.mean(dim=0, keepdim=True).expand_as(b))
                    else:
                        new.append(torch.zeros_like(b))
                else:
                    new.append(b)
            out.append(new)
        return out

    def _encode(self, text, visual, audio, fm, am):
        h_t, tok_t = self.enc_text(text)
        h_v, tok_v = self.enc_visual(visual, fm)
        h_a, tok_a = self.enc_audio(audio, am)
        cm_tv = self._cross(self.co_tv, self.co_vt, tok_t, tok_v, None, fm,
                            self.enc_text.pool, self.enc_visual.pool)
        cm_ta = self._cross(self.co_ta, self.co_at, tok_t, tok_a, None, am,
                            self.enc_text.pool, self.enc_audio.pool)
        cm_va = self._cross(self.co_va, self.co_av, tok_v, tok_a, fm, am,
                            self.enc_visual.pool, self.enc_audio.pool)
        logits_t = self.cls_text(h_t)
        logits_v = self.cls_visual(h_v)
        logits_a = self.cls_audio(h_a)
        return h_t, h_v, h_a, cm_tv, cm_ta, cm_va, logits_t, logits_v, logits_a

    @torch.no_grad()
    def _attribution_labels(self, blocks, gate_dispatch, label, raw=None):
        if self.attrib_ref == 'uniform':
            g = torch.full_like(gate_dispatch, 1.0 / self.num_experts)
        else:
            g = gate_dispatch.detach()
        logits_full = self._expert_logits(blocks)
        z_full = torch.einsum('bk,bk->b', g, logits_full[..., 1])
        s = []
        for m in ('t', 'v', 'a'):
            if self.attrib_mode in ('full', 'selfmean') and raw is not None:
                text, visual, audio, fm, am = raw
                text_m, vis_m, aud_m = text, visual, audio
                if self.attrib_mode == 'full':
                    if m == 't':
                        text_m = torch.zeros_like(text)
                    elif m == 'v':
                        vis_m = torch.zeros_like(visual)
                    else:
                        aud_m = torch.zeros_like(audio)
                else:
                    if m == 't':
                        text_m = text.mean(dim=1, keepdim=True).expand_as(text)
                    elif m == 'v':
                        vis_m = visual.mean(dim=1, keepdim=True).expand_as(visual)
                    else:
                        aud_m = audio.mean(dim=1, keepdim=True).expand_as(audio)
                content = self._encode(text_m, vis_m, aud_m, fm, am)
                blocks_m = self._blocks(content[0], content[1], content[2],
                                        content[3], content[4], content[5])
            else:
                blocks_m = self._ablated_blocks(blocks, m)
            logits_m = self._expert_logits(blocks_m)
            s.append(z_full - torch.einsum('bk,bk->b', g, logits_m[..., 1]))
        s = torch.stack(s, dim=1)
        target = torch.full_like(label, 4)
        fake = (label == 1)
        if fake.any():
            sf = s[fake]
            order = torch.argsort(sf, dim=1, descending=True)
            s1 = sf.gather(1, order[:, :1]).squeeze(1)
            s2 = sf.gather(1, order[:, 1:2]).squeeze(1)
            arg = order[:, 0]
            if self.attrib_relative > 0:
                exclusive = s1 >= self.attrib_relative * torch.clamp(s2, min=1e-6)
            else:
                exclusive = (s1 - s2) > self.attrib_margin
            single = (s1 > 0) & exclusive
            tgt = torch.where(single, arg, torch.full_like(arg, 3))
            target[fake] = tgt
        return target, s

    def forward(self, **kwargs):
        text = kwargs['text_fea']
        visual = kwargs['frames']
        audio = kwargs['audio_feas']
        label = kwargs['label']
        frames_masks = kwargs.get('frames_masks')
        audiofeas_masks = kwargs.get('audiofeas_masks')

        h_t, h_v, h_a, cm_tv, cm_ta, cm_va, logits_t, logits_v, logits_a = self._encode(
            text, visual, audio, frames_masks, audiofeas_masks)
        probs_t, probs_v, probs_a = (F.softmax(logits_t, -1), F.softmax(logits_v, -1),
                                     F.softmax(logits_a, -1))

        gate_in = [probs_t[:, 1:2], probs_v[:, 1:2], probs_a[:, 1:2]]   # P(fake | modality)
        if self.cm_proj is not None:
            gate_in.append(self.cm_proj(torch.cat([cm_tv, cm_ta, cm_va], dim=-1)))
        gate_raw = self.gate(torch.cat(gate_in, dim=-1))
        gate_dispatch = F.softmax(gate_raw, dim=-1)
        if self.force_uniform:
            gate_dispatch = torch.full_like(gate_dispatch, 1.0 / self.num_experts)

        blocks = self._blocks(h_t, h_v, h_a, cm_tv, cm_ta, cm_va)
        expert_logits = self._expert_logits(blocks)
        if self.loss_type == 'bce':
            output = torch.einsum('bk,bk->b', gate_dispatch, expert_logits.squeeze(-1)).unsqueeze(-1)
        else:
            output = torch.einsum('bk,bkc->bc', gate_dispatch, expert_logits)

        single_loss = (self.plain_ce(logits_t, label) + self.plain_ce(logits_v, label)
                       + self.plain_ce(logits_a, label)) / 3.0

        interaction_loss = torch.zeros((), device=label.device)
        if self.use_attrib and self.attrib_enabled:
            labels_fixed = self._offline_labels if self._offline_labels is not None else self.offline_labels
            if labels_fixed is not None:
                idx = kwargs['index'].to(labels_fixed.device)
                target = labels_fixed[idx].to(gate_raw.device)
            else:
                target, _ = self._attribution_labels(
                    blocks, gate_dispatch, label,
                    raw=(text, visual, audio, frames_masks, audiofeas_masks))
            weight = None
            if self.gate_weight != 'none':
                with torch.no_grad():
                    hist = torch.bincount(target, minlength=self.num_experts).float()
                    self.gate_counts.mul_(1.0 - self.gate_ema).add_(self.gate_ema * hist)
                    w = 1.0 / self.gate_counts.clamp(min=1e-6)
                    if self.gate_weight == 'sqrt_inv':
                        w = w.sqrt()
                    weight = (w / w.sum() * self.num_experts).detach()
            interaction_loss = F.cross_entropy(gate_raw, target, weight=weight)

        aux_loss = self.lambda_single * single_loss + self.lambda_interaction * interaction_loss
        return output, aux_loss


    @torch.no_grad()
    def diagnose(self, **kwargs):
        text, visual, audio = kwargs['text_fea'], kwargs['frames'], kwargs['audio_feas']
        label = kwargs['label']
        fm, am = kwargs.get('frames_masks'), kwargs.get('audiofeas_masks')

        h_t, h_v, h_a, cm_tv, cm_ta, cm_va, logits_t, logits_v, logits_a = self._encode(
            text, visual, audio, fm, am)
        gate_in = [F.softmax(logits_t, -1)[:, 1:2], F.softmax(logits_v, -1)[:, 1:2],
                   F.softmax(logits_a, -1)[:, 1:2]]
        if self.cm_proj is not None:
            gate_in.append(self.cm_proj(torch.cat([cm_tv, cm_ta, cm_va], dim=-1)))
        gate_raw = self.gate(torch.cat(gate_in, dim=-1))
        gate_dispatch = F.softmax(gate_raw, dim=-1)
        if self.force_uniform:
            gate_dispatch = torch.full_like(gate_dispatch, 1.0 / self.num_experts)
        blocks = self._blocks(h_t, h_v, h_a, cm_tv, cm_ta, cm_va)
        expert_logits = self._expert_logits(blocks)
        output = torch.einsum('bk,bkc->bc', gate_dispatch, expert_logits)
        target, s = self._attribution_labels(blocks, gate_dispatch, label,
                                             raw=(text, visual, audio, fm, am))
        return dict(output=output, probs=F.softmax(output, -1)[:, 1], gate=gate_dispatch,
                    target=target, s=s, expert_logits=expert_logits,
                    p_t=F.softmax(logits_t, -1)[:, 1], p_v=F.softmax(logits_v, -1)[:, 1],
                    p_a=F.softmax(logits_a, -1)[:, 1])
