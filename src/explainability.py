"""
src/mk_ensemble/explainability.py
SAC（Structure-Activity Contribution）可解释性模块

方法：Integrated Gradients（积分梯度）
  - 基线：零向量（空Fragment嵌入）
  - 积分步数：50步
  - 贡献度：Fragment嵌入维度对预测值的归因之和

用法：
    from src.mk_ensemble.explainability import fragment_ig_attribution
    attribs = fragment_ig_attribution(model, frag_emb, mol_idx, task_idx=0, n_steps=50)
    # attribs: [n_frags]  每个Fragment对task_idx任务预测的贡献度
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F


def _forward_from_emb(model, frag_emb: torch.Tensor, mol_idx: torch.Tensor,
                       task_idx: int) -> torch.Tensor:
    """
    从Fragment嵌入开始做前向传播（跳过GIN），返回指定task的预测值 [n_mols]
    这样可以对frag_emb求梯度。
    """
    n_mols = int(mol_idx.max().item()) + 1

    # Router → weights
    weights, _ = model.router(frag_emb)                      # [n_frags, n_experts]

    # Experts
    expert_outs = torch.stack(
        [expert(frag_emb) for expert in model.experts], dim=1
    )  # [n_frags, n_experts, d_frag]

    # 加权聚合
    frag_moe_out = (weights.unsqueeze(-1) * expert_outs).sum(dim=1)  # [n_frags, d_frag]

    # Fragment → 分子
    mol_emb = torch.zeros(n_mols, frag_moe_out.shape[-1], device=frag_moe_out.device)
    mol_emb.scatter_add_(0,
        mol_idx.unsqueeze(-1).expand_as(frag_moe_out),
        frag_moe_out,
    )
    frag_count = torch.zeros(n_mols, device=frag_moe_out.device)
    frag_count.scatter_add_(0, mol_idx,
                            torch.ones(len(mol_idx), device=frag_moe_out.device))
    mol_emb = mol_emb / frag_count.unsqueeze(-1).clamp(min=1)

    # 预测头
    pred = model.heads[task_idx](mol_emb).squeeze(-1)  # [n_mols]
    return pred


def fragment_ig_attribution(
    model,
    frag_emb: torch.Tensor,
    mol_idx: torch.Tensor,
    task_idx: int = 0,
    n_steps: int = 50,
) -> np.ndarray:
    """
    Integrated Gradients：计算每个Fragment嵌入对指定任务预测的贡献度。

    参数：
        model:      MKEnsemble模型（eval mode）
        frag_emb:   [n_frags, d_frag]  GIN编码的Fragment嵌入（已计算，detach）
        mol_idx:    [n_frags]           每个Fragment属于哪个分子
        task_idx:   目标任务索引（0=DPPH, 1=ABTS, 2=FRAP）
        n_steps:    积分步数（越多越精确，50足够）

    返回：
        attribution: [n_frags]  每个Fragment的贡献度（正=提高活性, 负=降低活性）
    """
    model.eval()
    baseline = torch.zeros_like(frag_emb)  # 零向量基线

    grads_list = []
    alphas = np.linspace(0.0, 1.0, n_steps + 1)

    for alpha in alphas:
        x_alpha = (baseline + alpha * (frag_emb - baseline)).detach().requires_grad_(True)

        pred = _forward_from_emb(model, x_alpha, mol_idx, task_idx)
        score = pred.sum()  # 所有分子预测之和（对梯度没有影响，只是标量化）

        grad = torch.autograd.grad(score, x_alpha, create_graph=False)[0]
        grads_list.append(grad.detach().cpu())

    # Riemann积分（梯形法）
    grads_tensor = torch.stack(grads_list, dim=0)  # [n_steps+1, n_frags, d_frag]
    avg_grad = grads_tensor.mean(dim=0)            # [n_frags, d_frag]

    # attribution = (x - baseline) · avg_grad，对d_frag维度求和
    delta       = (frag_emb - baseline).detach().cpu()
    attribution = (delta * avg_grad).sum(dim=-1).numpy()  # [n_frags]

    return attribution


def compute_all_attributions(
    model,
    smiles_list: list[str],
    task_names: list[str],
    device: str = "cpu",
    n_steps: int = 50,
) -> dict:
    """
    对所有分子的所有任务计算Fragment IG归因。

    返回：
        {
          "mol_idx_flat": [总Fragment数],       # 每个frag属于哪个mol
          "frag_smiles":  [总Fragment数],        # Fragment SMILES（近似）
          "attributions": {task: [总Fragment数]} # 各任务的贡献度
        }
    """
    from .fragment import smiles_to_fragments, decompose_molecule
    from torch_geometric.data import Batch

    all_frags_list = smiles_to_fragments(smiles_list)

    all_frags = []
    mol_idx_flat = []
    frag_smiles  = []

    for mol_i, frags in enumerate(all_frags_list):
        if not frags:
            continue
        all_frags.extend(frags)
        mol_idx_flat.extend([mol_i] * len(frags))
        # Fragment SMILES 近似（用原子数标记）
        for f in frags:
            frag_smiles.append(f"frag_mol{mol_i}_n{f.num_nodes}")

    frag_batch = Batch.from_data_list(all_frags).to(device)
    mol_idx_t  = torch.tensor(mol_idx_flat, dtype=torch.long).to(device)

    # 计算GIN嵌入（一次，共享）
    model.eval()
    with torch.no_grad():
        frag_emb = model.gin(frag_batch.x, frag_batch.edge_index, frag_batch.batch)
    # frag_emb: [n_frags, d_frag]

    attributions = {}
    for task_i, task_name in enumerate(task_names):
        attr = fragment_ig_attribution(
            model, frag_emb, mol_idx_t, task_idx=task_i, n_steps=n_steps
        )
        attributions[task_name] = attr

    return {
        "mol_idx_flat": np.array(mol_idx_flat),
        "frag_smiles":  frag_smiles,
        "attributions": attributions,
    }
