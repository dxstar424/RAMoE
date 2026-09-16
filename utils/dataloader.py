import os
import pandas as pd

from torch.utils.data import Dataset
import torch
import numpy as np

DATA_ROOT = os.environ.get('RAMOE_DATA_ROOT', 'data')


class MFSVFNDDataset(Dataset):
    def __init__(self, path_vid, dataset, max_text_len=512, visual_path=None, tokenizer=None):
        self.dataset = dataset
        self.vid = []
        self.max_text_len = max_text_len
        self.tokenizer = tokenizer

        if self.dataset == 'fakesv':
            root = os.path.join(DATA_ROOT, 'Chinese', 'fakesv')
            self.data_complete = pd.read_json(os.path.join(root, 'data_complete.json'),
                                              orient='records', dtype=False, lines=True)
            self.data_complete = self.data_complete[self.data_complete['annotation'] != '辟谣']
        else:
            root = os.path.join(DATA_ROOT, 'English', 'fakett')
            meta = os.path.join(root, 'new_data.json')
            if not os.path.exists(meta):
                meta = os.path.join(root, 'data.json')
            self.data_complete = pd.read_json(meta, orient='records', dtype=False,
                                              lines=True)
        self.maefeapath = visual_path if visual_path else os.path.join(root, 'mae_fea')
        self.hubert_path = os.path.join(root, 'hubert_fea') + os.sep
        self.bert_feapath = os.path.join(root, 'bert_fea') + os.sep
        split_base = os.path.join(root, 'data-split') + os.sep

        if isinstance(path_vid, str):
            path_vid = [path_vid]
        for p in path_vid:
            with open(split_base + p, "r") as fr:
                for line in fr.readlines():
                    self.vid.append(line.strip())

        self.data = self.data_complete[self.data_complete.video_id.isin(self.vid)].copy()
        self.data['video_id'] = self.data['video_id'].astype('category')
        self.data['video_id'].cat.set_categories(self.vid)
        self.data.sort_values('video_id', ascending=True, inplace=True)
        self.data.reset_index(inplace=True)

    def __len__(self):
        return self.data.shape[0]

    @staticmethod
    def _txt(v):
        if v is None:
            return ''
        if isinstance(v, str):
            return v
        try:
            if pd.isna(v):
                return ''
        except Exception:
            pass
        return str(v)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        vid = item['video_id']

        if self.dataset == 'fakesv':
            label = 0 if item['annotation'] == '真' else 1
        else:
            label = 0 if item['annotation'] == 'real' else 1
        label = torch.tensor(label)

        if self.tokenizer is not None:
            if self.dataset == 'fakesv':
                raw = (self._txt(item.get('title')) + ' ' + self._txt(item.get('ocr'))
                       + '' + self._txt(item.get('keywords')))
            else:
                raw = (self._txt(item.get('description')) + ' ' + self._txt(item.get('recognize_ocr'))
                       + '' + self._txt(item.get('event')))
            enc = self.tokenizer(raw, max_length=self.max_text_len,
                                 padding='max_length', truncation=True, return_tensors='pt')
            title_inputid = enc['input_ids'].squeeze(0).long()
            title_mask = enc['attention_mask'].squeeze(0).long()
            text_fea = None
        else:
            title_inputid = title_mask = None
            text_fea = torch.load(self.bert_feapath + vid + '.pkl', map_location='cpu').squeeze(0)
            if self.max_text_len > 0:
                text_fea = text_fea[:self.max_text_len]

        audio_fea = torch.load(self.hubert_path + vid + '.pkl', map_location='cpu')
        audio_fea = torch.squeeze(audio_fea).float()
        if audio_fea.dim() == 1:
            audio_fea = audio_fea.unsqueeze(0)

        frames = torch.load(os.path.join(self.maefeapath, vid + '.pkl'), map_location='cpu')
        frames = frames.float()
        if frames.dim() == 1:
            frames = frames.unsqueeze(0)

        out = {
            'label': label,
            'audio_fea': audio_fea,
            'frames': frames,
            'index': torch.tensor(idx),
        }
        if text_fea is not None:
            out['text_fea'] = text_fea
        else:
            out['title_inputid'] = title_inputid
            out['title_mask'] = title_mask
        return out


def pad_sequence_to_len(seq_len, lst):
    attention_masks = []
    result = []
    for item in lst:
        item = item.float()
        if item.dim() == 1:
            item = item.unsqueeze(0)
        ori_len = item.shape[0]
        if ori_len >= seq_len:
            gap = ori_len // seq_len
            item = item[::gap][:seq_len]
            mask = np.ones((seq_len,))
        else:
            item = torch.cat((item, torch.zeros([seq_len - ori_len, item.shape[1]], dtype=torch.float)), dim=0)
            mask = np.append(np.ones(ori_len), np.zeros(seq_len - ori_len))
        result.append(item)
        attention_masks.append(torch.IntTensor(mask))
    return torch.stack(result), torch.stack(attention_masks)


def _collate_fn(batch, num_frames, num_audioframes):
    has_local_text = ('text_fea' in batch[0])
    if has_local_text:
        text_fea = torch.stack([item['text_fea'] for item in batch])
    else:
        title_inputid = torch.stack([item['title_inputid'] for item in batch])
        title_mask = torch.stack([item['title_mask'] for item in batch])

    frames = [item['frames'] for item in batch]
    frames, frames_masks = pad_sequence_to_len(num_frames, frames)

    audio_feas = [item['audio_fea'] for item in batch]
    audio_feas, audiofeas_masks = pad_sequence_to_len(num_audioframes, audio_feas)

    label = torch.stack([item['label'] for item in batch])

    out = {
        'label': label,
        'audio_feas': audio_feas,
        'audiofeas_masks': audiofeas_masks,
        'frames': frames,
        'frames_masks': frames_masks,
        'index': torch.stack([item['index'] for item in batch]),
    }
    if has_local_text:
        out['text_fea'] = text_fea
    else:
        out['title_inputid'] = title_inputid
        out['title_mask'] = title_mask
    return out


def fakesv_collate_fn(batch, num_frames=86):
    num_audioframes = 80
    return _collate_fn(batch, num_frames, num_audioframes)


def fakett_collate_fn(batch, num_frames=111):
    num_audioframes = 103
    return _collate_fn(batch, num_frames, num_audioframes)
