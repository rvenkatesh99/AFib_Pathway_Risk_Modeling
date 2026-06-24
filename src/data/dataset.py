"""
Dataset classes for pathway-based AFib risk modeling.

Input data format:
  - Pathway matrix: (N, K, T) — N individuals, K pathways, T features per pathway
    (or a 2D matrix (N, K) if T=1, i.e., one score per pathway)
  - Additional columns (e.g., PRS) are passed as part of the covariate matrix
  - Covariates: (N, C) — age, sex, ancestry PCs 1-10, optional PRS
  - Labels: (N,) — binary AFib outcome

The DataLoader returns dicts with keys:
  'pathway_features': (batch, K, T) float tensor
  'covariates':       (batch, C) float tensor
  'label':            (batch,) float tensor
  'sample_id':        list[str] (optional)
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


class PathwayDataset(Dataset):
    """
    Args:
        pathway_matrix: np.ndarray of shape (N, K) or (N, K, T)
        covariate_matrix: np.ndarray of shape (N, C)
        labels: np.ndarray of shape (N,)
        sample_ids: list[str] | None
    """

    def __init__(
        self,
        pathway_matrix: np.ndarray,
        covariate_matrix: np.ndarray,
        labels: np.ndarray,
        sample_ids=None,
    ):
        if pathway_matrix.ndim == 2:
            pathway_matrix = pathway_matrix[:, :, np.newaxis]  # (N, K, 1)

        assert pathway_matrix.shape[0] == covariate_matrix.shape[0] == labels.shape[0]

        self.pathway = torch.tensor(pathway_matrix, dtype=torch.float32)
        self.covariates = torch.tensor(covariate_matrix, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.sample_ids = sample_ids if sample_ids is not None else [str(i) for i in range(len(labels))]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "pathway_features": self.pathway[idx],
            "covariates": self.covariates[idx],
            "label": self.labels[idx],
            "sample_id": self.sample_ids[idx],
        }


def make_weighted_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    """Inverse-frequency weighted sampler for class imbalance."""
    class_counts = np.bincount(labels.astype(int))
    weights = 1.0 / class_counts[labels.astype(int)]
    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.float32),
        num_samples=len(labels),
        replacement=True,
    )


def make_dataloaders(
    train_pathway, train_cov, train_labels,
    val_pathway, val_cov, val_labels,
    test_pathway, test_cov, test_labels,
    batch_size: int = 256,
    num_workers: int = 4,
    train_sample_ids=None,
    val_sample_ids=None,
    test_sample_ids=None,
):
    train_ds = PathwayDataset(train_pathway, train_cov, train_labels, train_sample_ids)
    val_ds = PathwayDataset(val_pathway, val_cov, val_labels, val_sample_ids)
    test_ds = PathwayDataset(test_pathway, test_cov, test_labels, test_sample_ids)

    sampler = make_weighted_sampler(train_labels)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def load_data_from_files(
    pathway_matrix_path: str,
    covariate_path: str,
    label_col: str = "afib",
    covariate_cols: list = None,
) -> tuple:
    """
    Load pathway matrix and covariates from CSV/TSV/parquet files.

    pathway_matrix_path: file where rows=individuals, columns=pathway scores
                         (optionally includes a sample_id column)
    covariate_path: file with covariates; must have same row order as pathway matrix
    label_col: column name for AFib outcome in covariate file
    covariate_cols: columns to use as covariates; if None uses all except label_col and sample_id

    Returns (pathway_matrix, covariate_matrix, labels, pathway_names, covariate_names, sample_ids)
    """
    def _read(path):
        if path.endswith(".parquet"):
            return pd.read_parquet(path)
        sep = "\t" if path.endswith(".tsv") else ","
        return pd.read_csv(path, sep=sep)

    pw_df = _read(pathway_matrix_path)
    cov_df = _read(covariate_path)

    sample_id_col = "sample_id" if "sample_id" in pw_df.columns else None
    sample_ids = pw_df[sample_id_col].tolist() if sample_id_col else None
    pathway_cols = [c for c in pw_df.columns if c != sample_id_col]
    pathway_matrix = pw_df[pathway_cols].values.astype(np.float32)
    pathway_names = pathway_cols

    labels = cov_df[label_col].values.astype(np.float32)
    if covariate_cols is None:
        covariate_cols = [c for c in cov_df.columns if c not in (label_col, "sample_id")]
    covariate_matrix = cov_df[covariate_cols].values.astype(np.float32)

    return pathway_matrix, covariate_matrix, labels, pathway_names, covariate_cols, sample_ids
