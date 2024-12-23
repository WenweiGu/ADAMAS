import numpy as np
from torch.utils.data import Dataset
import torch
from sklearn.preprocessing import minmax_scale


def Anomaly_Removal(train_data):
    train_metrics = np.array(train_data[0])
    train_labels = np.array(train_data[1])

    normal_metrics = train_metrics[train_labels == 0]
    normal_labels = train_labels[train_labels == 0]

    return [normal_metrics, normal_labels]


def Normal_Period(train_data):
    train_metrics = train_data[0]
    train_labels = train_data[1]

    max_length = 0
    start_index = None
    end_index = None
    current_length = 0
    current_start_index = None

    for idx, value in enumerate(train_labels):
        if value is False or value == 0:
            if current_length == 0:
                current_start_index = idx
            current_length += 1
        else:
            if current_length > max_length:
                max_length = current_length
                start_index = current_start_index
                end_index = idx - 1
            current_length = 0

    # Check if the last sequence is the longest
    if current_length > max_length:
        start_index = current_start_index
        end_index = len(train_labels) - 1

    normal_metrics = train_metrics[start_index: end_index]
    normal_labels = train_labels[start_index: end_index]

    return [normal_metrics, normal_labels]


class UTSDataset(Dataset):
    def __init__(self, raw_seqs, labels, win_len=20, minmax=True) -> None:
        super().__init__()

        if minmax:
            raw_seqs = minmax_scale(raw_seqs)

        self._raw_seqs = torch.tensor(raw_seqs, dtype=torch.float32)
        self._labels = torch.tensor(labels, dtype=torch.bool)
        self._win_len = win_len

    def __len__(self):
        return len(self._raw_seqs) - self._win_len + 1
    
    def __getitem__(self, index):
        return self._raw_seqs[index: index + self._win_len], self._labels[index + self._win_len - 1]

    def set_win_len(self, win_len):
        self._win_len = win_len


class AIOPS18Dataset(UTSDataset):
    def __init__(self, kpi, win_len=20, minmax=True) -> None:
        raw_seq = kpi[0]
        labels = kpi[1]

        super().__init__(raw_seq, labels, win_len, minmax)
