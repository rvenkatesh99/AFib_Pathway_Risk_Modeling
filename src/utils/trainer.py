"""
Neural model trainer with:
  - AdamW optimizer + cosine annealing LR schedule
  - Weighted binary cross-entropy for class imbalance
  - Early stopping on validation AUROC (patience=15)
  - Hyperparameter tuning interface
"""

import copy
import math
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score


class EarlyStopping:
    def __init__(self, patience: int = 15, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = -np.inf
        self.counter = 0
        self.best_state = None

    def step(self, score: float, model: nn.Module) -> bool:
        """Returns True if training should stop."""
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model: nn.Module):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def compute_class_weights(labels: np.ndarray, device: str = "cpu") -> torch.Tensor:
    """Inverse frequency class weights for BCEWithLogitsLoss pos_weight."""
    n_neg = (labels == 0).sum()
    n_pos = (labels == 1).sum()
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)
    return pos_weight


def train_epoch(model, loader, optimizer, criterion, device: str):
    model.train()
    total_loss = 0.0
    for batch in loader:
        pw = batch["pathway_features"].to(device)
        cov = batch["covariates"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(pw, cov)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device: str):
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []
    for batch in loader:
        pw = batch["pathway_features"].to(device)
        cov = batch["covariates"].to(device)
        labels = batch["label"].to(device)

        logits = model(pw, cov)
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)

        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    auroc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5
    return total_loss / len(loader.dataset), auroc, all_probs, all_labels


def train(
    model: nn.Module,
    train_loader,
    val_loader,
    train_labels: np.ndarray,
    n_epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 15,
    device: str = "cpu",
    verbose: bool = True,
):
    """
    Full training loop.
    Returns (trained_model, training_history_dict).
    """
    model = model.to(device)
    pos_weight = compute_class_weights(train_labels, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr / 100)
    stopper = EarlyStopping(patience=patience)

    history = {"train_loss": [], "val_loss": [], "val_auroc": []}

    for epoch in range(1, n_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_auroc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auroc"].append(val_auroc)

        if verbose and epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_auroc={val_auroc:.4f}")

        if stopper.step(val_auroc, model):
            if verbose:
                print(f"Early stopping at epoch {epoch} (best val_auroc={stopper.best_score:.4f})")
            break

    stopper.restore_best(model)
    return model, history


def tune_hyperparameters(
    model_cls,
    model_kwargs_grid: list,
    train_loader,
    val_loader,
    train_labels: np.ndarray,
    device: str = "cpu",
    n_epochs: int = 100,
    patience: int = 10,
) -> tuple:
    """
    Grid search over model hyperparameters evaluated on the validation set.

    model_kwargs_grid: list of dicts, each containing the full set of kwargs
                       for model_cls. Training hyperparameters (lr, weight_decay)
                       should be included in each dict and are popped before
                       passing to the model constructor.

    Returns (best_model_kwargs, best_train_kwargs, best_val_auroc, results_list).
    results_list is sorted descending by val_auroc for logging.
    """
    TRAIN_KEYS = {"lr", "weight_decay"}

    best_auroc = -1.0
    best_model_kwargs = None
    best_train_kwargs = None
    results = []

    for i, kwargs in enumerate(model_kwargs_grid):
        train_kwargs = {k: v for k, v in kwargs.items() if k in TRAIN_KEYS}
        model_kwargs = {k: v for k, v in kwargs.items() if k not in TRAIN_KEYS}

        model = model_cls(**model_kwargs)
        _, history = train(
            model, train_loader, val_loader, train_labels,
            n_epochs=n_epochs, patience=patience, device=device, verbose=False,
            **train_kwargs,
        )
        auroc = max(history["val_auroc"])
        results.append({"kwargs": kwargs, "val_auroc": auroc})
        print(f"  [{i+1}/{len(model_kwargs_grid)}] val_auroc={auroc:.4f} | {kwargs}")

        if auroc > best_auroc:
            best_auroc = auroc
            best_model_kwargs = model_kwargs
            best_train_kwargs = train_kwargs

    results.sort(key=lambda x: -x["val_auroc"])
    return best_model_kwargs, best_train_kwargs, best_auroc, results
