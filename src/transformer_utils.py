import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.encodings = tokenizer(
            texts, truncation=True, padding='max_length',
            max_length=max_len, return_tensors='pt'
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids':      self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels':         self.labels[idx]
        }


class InferenceDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=max_len,
            return_tensors='pt'
        )

    def __len__(self):
        return self.encodings['input_ids'].shape[0]

    def __getitem__(self, idx):
        return {
            'input_ids':      self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx]
        }


def label_smoothing_loss(logits, labels, n_classes, smoothing=0.1, weights=None):
    log_probs     = F.log_softmax(logits, dim=-1)
    smooth_target = torch.full_like(log_probs, smoothing / (n_classes - 1))
    smooth_target.scatter_(1, labels.unsqueeze(1), 1.0 - smoothing)
    loss = -(smooth_target * log_probs).sum(dim=-1)
    if weights is not None:
        loss = loss * weights[labels]
    return loss.mean()
