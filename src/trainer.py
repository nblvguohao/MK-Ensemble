"""
src/mk_ensemble/trainer.py
MKEnsemble 训练器

- 多任务回归（DPPH / ABTS / FRAP）
- LOOCV 评估（数据量 < 20 时自动切换）
- 辅助负载均衡损失
- 早停 + CosineAnnealingLR
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Batch
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from .fragment import smiles_to_fragments
from .model import MKEnsemble

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_batch(smiles_list: list[str], targets: np.ndarray, indices: list[int]
                ) -> tuple[Batch, torch.Tensor, torch.Tensor]:
    """
    为指定indices的样本构建 PyG Batch

    返回:
      frag_batch: PyG Batch（所有Fragment）
      mol_idx:    [n_frags] 每个Fragment属于哪个分子（0-indexed in batch）
      y:          [n_mols, n_tasks]
    """
    all_frags_list = smiles_to_fragments([smiles_list[i] for i in indices])

    all_frags = []
    mol_idx   = []
    for batch_i, frags in enumerate(all_frags_list):
        if not frags:
            continue
        all_frags.extend(frags)
        mol_idx.extend([batch_i] * len(frags))

    frag_batch = Batch.from_data_list(all_frags).to(DEVICE)
    mol_idx_t  = torch.tensor(mol_idx, dtype=torch.long).to(DEVICE)
    y          = torch.tensor(targets[indices], dtype=torch.float).to(DEVICE)

    return frag_batch, mol_idx_t, y


def train_epoch(model: MKEnsemble, optimizer, smiles_list, targets, train_idx):
    """单epoch训练，返回平均loss"""
    model.train()
    frag_batch, mol_idx_t, y = build_batch(smiles_list, targets, train_idx)

    optimizer.zero_grad()
    preds, lb_loss = model(frag_batch, mol_idx_t)

    # 多任务MSE损失（忽略NaN）
    mask = ~torch.isnan(y)
    task_loss = F.mse_loss(preds[mask], y[mask]) if mask.any() else torch.tensor(0.0)

    loss = task_loss + model.lb_coef * lb_loss
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()

    return float(loss.item())


import torch.nn.functional as F


def evaluate(model: MKEnsemble, smiles_list, targets, val_idx):
    """评估，返回每任务的预测值"""
    model.eval()
    with torch.no_grad():
        frag_batch, mol_idx_t, y = build_batch(smiles_list, targets, val_idx)
        preds, _ = model(frag_batch, mol_idx_t)
    return preds.cpu().numpy(), y.cpu().numpy()


def compute_metrics(y_true, y_pred):
    metrics = {}
    for task_i in range(y_true.shape[1] if y_true.ndim > 1 else 1):
        yt = y_true[:, task_i] if y_true.ndim > 1 else y_true
        yp = y_pred[:, task_i] if y_pred.ndim > 1 else y_pred
        mask = ~np.isnan(yt)
        if mask.sum() < 2:
            metrics[f"task_{task_i}"] = {"R2": float("nan")}
            continue
        yt, yp = yt[mask], yp[mask]
        r2   = r2_score(yt, yp)
        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        mae  = float(mean_absolute_error(yt, yp))
        r, _ = pearsonr(yt, yp)
        metrics[f"task_{task_i}"] = {"R2": r2, "RMSE": rmse, "MAE": mae, "Pearson_r": float(r)}
    return metrics


def run_loocv(smiles_list, targets, task_names,
              max_epochs=200, patience=20, lr=1e-3, weight_decay=1e-4,
              d_frag=128, n_experts=8, expert_hid=256, n_gin=3, top_k=2,
              seed=42, verbose=True):
    """
    LOOCV训练评估

    targets: [n_mols, n_tasks]  可包含NaN
    返回: all_true [n_mols, n_tasks], all_pred [n_mols, n_tasks]
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    n = len(smiles_list)
    all_true = np.full((n, len(task_names)), np.nan)
    all_pred = np.full((n, len(task_names)), np.nan)

    for leave_out in range(n):
        train_idx = [i for i in range(n) if i != leave_out]
        val_idx   = [leave_out]

        # 初始化新模型
        model = MKEnsemble(
            n_tasks=len(task_names),
            d_frag=d_frag,
            n_experts=n_experts,
            expert_hid=expert_hid,
            n_gin=n_gin,
            top_k=top_k,
        ).to(DEVICE)

        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)

        best_train_loss = float("inf")
        no_improve = 0

        for epoch in range(max_epochs):
            loss = train_epoch(model, optimizer, smiles_list, targets, train_idx)
            scheduler.step()

            # 早停：基于训练损失（LOOCV无验证集）
            if loss < best_train_loss - 1e-4:
                best_train_loss = loss
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                if verbose:
                    print(f"    LOO-{leave_out}: 早停 epoch={epoch+1}, loss={loss:.4f}")
                break

        # 预测留出点
        pred, true = evaluate(model, smiles_list, targets, val_idx)
        all_pred[leave_out] = pred[0]
        all_true[leave_out] = true[0]

        if verbose:
            print(f"  LOO {leave_out+1:2d}/{n}: loss={best_train_loss:.4f}  "
                  f"pred={pred[0]}  true={true[0]}")

    return all_true, all_pred


def run_kfold_cv(smiles_list, targets, task_names,
                 n_folds=5,
                 max_epochs=200, patience=20, lr=1e-3, weight_decay=1e-4,
                 d_frag=128, n_experts=8, expert_hid=256, n_gin=3, top_k=2,
                 seed=42, verbose=True):
    """
    K折交叉验证（n >= 20 时使用，比LOOCV快得多）

    targets: [n_mols, n_tasks]  可包含NaN
    返回: all_true [n_mols, n_tasks], all_pred [n_mols, n_tasks]
    """
    from sklearn.model_selection import KFold

    torch.manual_seed(seed)
    np.random.seed(seed)

    n = len(smiles_list)
    all_true = np.full((n, len(task_names)), np.nan)
    all_pred = np.full((n, len(task_names)), np.nan)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for fold_i, (train_idx, val_idx) in enumerate(kf.split(range(n))):
        if verbose:
            print(f"\n  Fold {fold_i+1}/{n_folds}: train={len(train_idx)}, val={len(val_idx)}")

        model = MKEnsemble(
            n_tasks=len(task_names),
            d_frag=d_frag,
            n_experts=n_experts,
            expert_hid=expert_hid,
            n_gin=n_gin,
            top_k=top_k,
        ).to(DEVICE)

        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)

        best_val_loss  = float("inf")
        best_state     = None
        no_improve     = 0

        for epoch in range(max_epochs):
            train_loss = train_epoch(model, optimizer, smiles_list, targets, list(train_idx))
            scheduler.step()

            # 用验证集上的MSE作为早停依据（k折有验证集）
            model.eval()
            with torch.no_grad():
                frag_batch, mol_idx_t, y_val = build_batch(
                    smiles_list, targets, list(val_idx))
                preds_val, _ = model(frag_batch, mol_idx_t)
                mask = ~torch.isnan(y_val)
                val_loss = float(F.mse_loss(preds_val[mask], y_val[mask]).item()) \
                    if mask.any() else float("inf")

            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_state    = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve    = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                if verbose:
                    print(f"    早停 epoch={epoch+1}, val_loss={val_loss:.4f}")
                break

        # 用最佳权重预测
        if best_state is not None:
            model.load_state_dict(best_state)

        pred, true = evaluate(model, smiles_list, targets, list(val_idx))
        for j, vi in enumerate(val_idx):
            all_pred[vi] = pred[j]
            all_true[vi] = true[j]

        if verbose:
            print(f"  Fold {fold_i+1} best_val_loss={best_val_loss:.4f}")

    return all_true, all_pred
