"""
Dataset utilities for flow-based sequence creation and multi-task data loading.

Transforms tabular network flow data into temporal sequences suitable for
transformer-based DDoS detection.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def create_flow_based_sequences(df, labels, seq_len=64, feature_cols=None):
    """
    Transform tabular network flows into temporal sequences.

    Groups flows by protocol type, sorts by timestamp, and creates sliding
    windows of consecutive flows with 50% overlap.

    Args:
        df: DataFrame with flow features (must include 'Protocol' and 'Timestamp' columns).
        labels: Array of encoded integer labels, one per flow.
        seq_len: Number of consecutive flows per sequence (default: 64).
        feature_cols: List of feature column names. If None, auto-detected
                      by excluding Protocol, Timestamp, label, and original_index.

    Returns:
        sequences: np.ndarray of shape (N, seq_len, feature_dim).
        sequence_labels: np.ndarray of shape (N,) — majority-vote label per sequence.
        protocol_seqs: np.ndarray of shape (N, seq_len) — protocol value per flow.
        temporal_seqs: np.ndarray of shape (N, seq_len) — normalized timestamp per flow.
    """
    df_combined = df.copy()
    df_combined['label'] = labels

    if 'Timestamp' in df_combined.columns:
        df_combined = df_combined.sort_values('Timestamp').reset_index(drop=True)

    if feature_cols is None:
        exclude = {'Timestamp', 'Protocol', 'label', 'original_index'}
        feature_cols = [c for c in df.columns if c not in exclude]

    sequences = []
    sequence_labels = []
    protocol_seqs = []
    temporal_seqs = []

    if 'Protocol' in df_combined.columns:
        protocol_groups = df_combined.groupby('Protocol')
    else:
        protocol_groups = [('unknown', df_combined)]

    stride = seq_len // 2  # 50% overlap

    for _protocol, group in protocol_groups:
        if 'Timestamp' in group.columns:
            group = group.sort_values('Timestamp').reset_index(drop=True)

        for start in range(0, len(group) - seq_len + 1, stride):
            window = group.iloc[start:start + seq_len]

            # Features
            sequences.append(window[feature_cols].values.astype(np.float32))

            # Protocol
            if 'Protocol' in window.columns:
                protocol_seqs.append(window['Protocol'].values.astype(np.float32))
            else:
                protocol_seqs.append(np.zeros(seq_len, dtype=np.float32))

            # Temporal (normalized within window)
            if 'Timestamp' in window.columns:
                ts = window['Timestamp'].values
                if ts.dtype == 'object':
                    try:
                        ts = pd.to_datetime(ts).astype(np.int64) / 1e9
                    except Exception:
                        ts = np.arange(seq_len, dtype=np.float32)
                else:
                    ts = ts.astype(np.float32)
                if len(np.unique(ts)) > 1:
                    ts = (ts - ts.min()) / (ts.max() - ts.min() + 1e-8)
                else:
                    ts = np.zeros_like(ts, dtype=np.float32)
            else:
                ts = np.arange(seq_len, dtype=np.float32) / seq_len
            temporal_seqs.append(ts.astype(np.float32))

            # Label (majority vote)
            labs = window['label'].values
            unique, counts = np.unique(labs, return_counts=True)
            sequence_labels.append(unique[counts.argmax()])

    return (
        np.array(sequences),
        np.array(sequence_labels),
        np.array(protocol_seqs),
        np.array(temporal_seqs),
    )


def create_binary_labels(multiclass_labels, label_encoder):
    """Convert multiclass labels to binary: 0 = benign, 1 = attack."""
    benign_idx = None
    for i, name in enumerate(label_encoder.classes_):
        if 'BENIGN' in str(name).upper() or 'NORMAL' in str(name).upper():
            benign_idx = i
            break
    if benign_idx is None:
        benign_idx = 0
    return (multiclass_labels != benign_idx).astype(np.int64)


def create_anomaly_scores(multiclass_labels, label_encoder):
    """Create continuous anomaly scores based on attack type severity."""
    scores = np.zeros(len(multiclass_labels), dtype=np.float32)
    for i, name in enumerate(label_encoder.classes_):
        upper = str(name).upper()
        if 'BENIGN' in upper or 'NORMAL' in upper:
            severity = 0.0
        elif 'SYN' in upper or 'TCP' in upper:
            severity = 0.9
        elif 'UDP' in upper:
            severity = 0.8
        elif 'HTTP' in upper or 'SLOWLORIS' in upper:
            severity = 0.7
        elif 'DNS' in upper or 'NTP' in upper:
            severity = 0.6
        else:
            severity = 0.5
        scores[multiclass_labels == i] = severity
    return scores


class FlowSequenceDataset(Dataset):
    """Single-task dataset for multiclass classification."""

    def __init__(self, features, protocols, temporal, labels):
        self.features = torch.from_numpy(features).float()
        self.protocols = torch.from_numpy(protocols).float()
        self.temporal = torch.from_numpy(temporal).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'protocols': self.protocols[idx],
            'temporal': self.temporal[idx],
            'labels': self.labels[idx],
        }


class MultiTaskDataset(Dataset):
    """Multi-task dataset with binary, multiclass, and anomaly labels."""

    def __init__(self, features, protocols, temporal,
                 multiclass_labels, binary_labels, anomaly_scores):
        self.features = torch.from_numpy(features).float()
        self.protocols = torch.from_numpy(protocols).float()
        self.temporal = torch.from_numpy(temporal).float()
        self.multiclass_labels = torch.from_numpy(multiclass_labels).long()
        self.binary_labels = torch.from_numpy(binary_labels).long()
        self.anomaly_scores = torch.from_numpy(anomaly_scores).float()

    def __len__(self):
        return len(self.multiclass_labels)

    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'protocols': self.protocols[idx],
            'temporal': self.temporal[idx],
            'multiclass_labels': self.multiclass_labels[idx],
            'binary_labels': self.binary_labels[idx],
            'anomaly_scores': self.anomaly_scores[idx],
        }
