import copy
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

def load_data(pathway_matrix_path, covariate_path, label_col="afib", covariate_cols=None):
    def _read(path):
        if path.endswith(".parquet"):
            return pd.read_parquet(path)
        return pd.read_csv(path, sep="\t" if path.endswith(".tsv") else ",")

    pw_df  = _read(pathway_matrix_path)
    cov_df = _read(covariate_path)

    id_col      = "sample_id" if "sample_id" in pw_df.columns else None
    pw_cols     = [c for c in pw_df.columns if c != id_col]
    pw          = pw_df[pw_cols].values.astype(np.float32)
    sample_ids  = pw_df[id_col].tolist() if id_col else None

    labels = cov_df[label_col].values.astype(np.float32)
    if covariate_cols is None:
        covariate_cols = [c for c in cov_df.columns if c not in (label_col, "sample_id")]
    cov = cov_df[covariate_cols].values.astype(np.float32)

    return pw, cov, labels, pw_cols, covariate_cols, sample_ids

class PathwayDataset(Dataset):
    def __init__(self, pw, cov, labels):
        if pw.ndim == 2:
            pw = pw[:, :, np.newaxis]
        self.pw     = torch.tensor(pw,     dtype=torch.float32)
        self.cov    = torch.tensor(cov,    dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {"pathway_features": self.pw[idx],
                "covariates":       self.cov[idx],
                "label":            self.labels[idx]}

def make_loaders(pw_tr, cov_tr, y_tr, pw_va, cov_va, y_va, pw_te, cov_te, y_te, batch_size=256):
    counts  = np.bincount(y_tr.astype(int))
    weights = torch.tensor(1.0 / counts[y_tr.astype(int)], dtype=torch.float32)
    sampler = WeightedRandomSampler(weights, num_samples=len(y_tr), replacement=True)

    tr = DataLoader(PathwayDataset(pw_tr, cov_tr, y_tr), batch_size=batch_size,
                    sampler=sampler, num_workers=0)
    va = DataLoader(PathwayDataset(pw_va, cov_va, y_va), batch_size=batch_size,
                    shuffle=False, num_workers=0)
    te = DataLoader(PathwayDataset(pw_te, cov_te, y_te), batch_size=batch_size,
                    shuffle=False, num_workers=0)
    return tr, va, te

class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_score = -np.inf
        self.counter    = 0
        self.best_state = None

    def step(self, score, model):
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter    = 0
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model):
        model.load_state_dict(self.best_state)

def _focal_loss(logits, targets, gamma=2.0):
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt  = torch.exp(-bce)
    return ((1 - pt) ** gamma * bce).mean()

def train(model, train_loader, val_loader, train_labels,
          n_epochs=200, lr=1e-3, weight_decay=1e-4, patience=15, device="cpu", verbose=True):
    model     = model.to(device)
    criterion = _focal_loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr/100)
    stopper   = EarlyStopping(patience=patience)
    history   = {"train_loss": [], "val_loss": [], "val_auroc": []}
    t0        = time.time()

    for epoch in range(1, n_epochs + 1):
        model.train()
        total = 0.0
        for batch in train_loader:
            pw, cov, y = (batch[k].to(device) for k in ("pathway_features", "covariates", "label"))
            optimizer.zero_grad()
            loss = criterion(model(pw, cov), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * y.size(0)
        train_loss = total / len(train_loader.dataset)

        val_loss, val_auroc = _evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auroc"].append(val_auroc)

        if verbose and epoch % 10 == 0:
            elapsed = time.time() - t0
            print(f"  epoch {epoch:3d}/{n_epochs} | "
                  f"train={train_loss:.4f} | val={val_loss:.4f} | auroc={val_auroc:.4f} | "
                  f"{elapsed:.0f}s elapsed", flush=True)

        if stopper.step(val_auroc, model):
            if verbose:
                elapsed = time.time() - t0
                print(f"  early stop at epoch {epoch} | "
                      f"best val auroc={stopper.best_score:.4f} | {elapsed:.0f}s elapsed", flush=True)
            break

    stopper.restore_best(model)
    return model, history

@torch.no_grad()
def _evaluate(model, loader, criterion, device):
    model.eval()
    total, probs, labels = 0.0, [], []
    for batch in loader:
        pw, cov, y = (batch[k].to(device) for k in ("pathway_features", "covariates", "label"))
        logits = model(pw, cov)
        total += criterion(logits, y).item() * y.size(0)
        probs.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(y.cpu().numpy())
    probs  = np.concatenate(probs)
    labels = np.concatenate(labels)
    auroc  = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5
    return total / len(loader.dataset), auroc

@torch.no_grad()
def _get_logits(model, loader, device):
    model.eval()
    logits_all, labels_all = [], []
    for batch in loader:
        pw, cov, y = (batch[k].to(device) for k in ("pathway_features", "covariates", "label"))
        logits_all.append(model(pw, cov).cpu().numpy())
        labels_all.append(y.cpu().numpy())
    return np.concatenate(logits_all), np.concatenate(labels_all)

def fit_platt_scaler(model, val_loader, device):
    logits, labels = _get_logits(model, val_loader, device)
    scaler = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    scaler.fit(logits.reshape(-1, 1), labels)
    return scaler

def predict_calibrated(model, loader, device, platt_scaler=None):
    logits, labels = _get_logits(model, loader, device)
    if platt_scaler is not None:
        probs = platt_scaler.predict_proba(logits.reshape(-1, 1))[:, 1]
    else:
        probs = 1 / (1 + np.exp(-logits))
    return probs, labels

def tune_hyperparameters(model_cls, model_kwargs_grid, train_loader, val_loader,
                         train_labels, device="cpu", n_epochs=60, patience=8):
    TRAIN_KEYS = {"lr", "weight_decay"}
    best_auroc, best_model_kw, best_train_kw, results = -1.0, None, None, []

    for i, kwargs in enumerate(model_kwargs_grid):
        t_cand = time.time()
        train_kw = {k: v for k, v in kwargs.items() if k in TRAIN_KEYS}
        model_kw = {k: v for k, v in kwargs.items() if k not in TRAIN_KEYS}
        _, history = train(model_cls(**model_kw), train_loader, val_loader, train_labels,
                           n_epochs=n_epochs, patience=patience, device=device,
                           verbose=False, **train_kw)
        auroc = max(history["val_auroc"])
        elapsed = time.time() - t_cand
        results.append({"kwargs": kwargs, "val_auroc": auroc})
        print(f"  [{i+1}/{len(model_kwargs_grid)}] auroc={auroc:.4f} | {elapsed:.0f}s | {kwargs}", flush=True)
        if auroc > best_auroc:
            best_auroc, best_model_kw, best_train_kw = auroc, model_kw, train_kw

    results.sort(key=lambda x: -x["val_auroc"])
    return best_model_kw, best_train_kw, best_auroc, results
