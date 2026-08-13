"""
src/mk_ensemble/model.py
MKEnsemble 核心模型

架构：
  SMILES
    → Fragment分解（Murcko + BRICS）
    → GIN编码器（per-fragment）  → d_frag=128
    → Router（Gating Network）  → top-k=2 软路由
    → MoE层（n_experts=8, hidden=256）
    → 加权聚合所有Fragment输出
    → 多任务预测头（DPPH / ABTS / FRAP）

设计原则（来自spec）：
- Router使用软路由（Softmax），不用argmax（防Expert collapse）
- 辅助负载均衡损失（load_balance_loss）
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.data import Data, Batch

from .fragment import ATOM_FEAT_DIM, BOND_FEAT_DIM

# ─── GIN 单Fragment编码器 ──────────────────────────────────────────────────────

class GINEncoder(nn.Module):
    """
    3层 GIN，输出 d_frag 维 embedding（graph-level）
    """
    def __init__(self, in_dim: int = ATOM_FEAT_DIM,
                 hidden_dim: int = 128, out_dim: int = 128,
                 n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()

        dims = [in_dim] + [hidden_dim] * (n_layers - 1) + [out_dim]
        for i in range(n_layers):
            mlp = nn.Sequential(
                nn.Linear(dims[i], dims[i+1]),
                nn.ReLU(),
                nn.Linear(dims[i+1], dims[i+1]),
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(dims[i+1]))

        self.out_dim = out_dim

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch: torch.Tensor) -> torch.Tensor:
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return global_mean_pool(x, batch)  # [batch_size, out_dim]

# ─── MoE Expert ───────────────────────────────────────────────────────────────

class Expert(nn.Module):
    """3层MLP Expert"""
    def __init__(self, in_dim: int, hidden_dim: int = 256, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,    hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# ─── Router（Gating Network）─────────────────────────────────────────────────

class Router(nn.Module):
    """
    软路由：输出每个Expert的权重（Softmax归一化）
    top-k=2：每个Fragment激活权重最高的2个Expert
    """
    def __init__(self, in_dim: int, n_experts: int = 8, k: int = 2,
                 noise_std: float = 0.1):
        super().__init__()
        self.k = k
        self.n_experts = n_experts
        self.noise_std = noise_std
        self.gate = nn.Linear(in_dim, n_experts, bias=False)

    def forward(self, x: torch.Tensor):
        """
        x: [n_frags, in_dim]
        返回:
          weights: [n_frags, n_experts]  各Expert的加权系数（稀疏，top-k非零）
          load:    [n_experts]           每个Expert的平均负载（用于负载均衡损失）
        """
        logits = self.gate(x)  # [n_frags, n_experts]

        # 训练时加噪声防止路由坍塌
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise

        # Top-k mask
        topk_vals, topk_idx = torch.topk(logits, self.k, dim=-1)
        mask = torch.full_like(logits, float("-inf"))
        mask.scatter_(-1, topk_idx, topk_vals)

        weights = F.softmax(mask, dim=-1)  # [n_frags, n_experts]
        load    = weights.mean(dim=0)      # [n_experts]

        return weights, load

# ─── MKEnsemble 主模型 ─────────────────────────────────────────────────────────

class MKEnsemble(nn.Module):
    """
    Fragment Multi-Kernel Ensemble 分子属性预测模型

    参数：
      n_tasks:    输出任务数（DPPH=1, ABTS=1, FRAP=1 → 3）
      d_frag:     Fragment embedding维度
      n_experts:  Expert数量
      expert_hid: Expert隐层维度
      n_gin:      GIN层数
      top_k:      每个Fragment激活的Expert数
      lb_coef:    负载均衡损失系数
    """
    def __init__(self,
                 n_tasks:    int   = 3,
                 d_frag:     int   = 128,
                 n_experts:  int   = 8,
                 expert_hid: int   = 256,
                 n_gin:      int   = 3,
                 top_k:      int   = 2,
                 lb_coef:    float = 0.01):
        super().__init__()
        self.lb_coef = lb_coef
        self.n_tasks = n_tasks

        # GIN编码器（共享参数，所有Fragment共用）
        self.gin = GINEncoder(
            in_dim=ATOM_FEAT_DIM,
            hidden_dim=d_frag,
            out_dim=d_frag,
            n_layers=n_gin,
        )

        # Router
        self.router = Router(d_frag, n_experts=n_experts, k=top_k)

        # Experts
        self.experts = nn.ModuleList([
            Expert(d_frag, hidden_dim=expert_hid, out_dim=d_frag)
            for _ in range(n_experts)
        ])

        # 多任务预测头（每个任务独立）
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_frag, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )
            for _ in range(n_tasks)
        ])

    def encode_fragments(self, frag_batch: Batch) -> torch.Tensor:
        """
        编码一个batch的Fragment图
        frag_batch: PyG Batch，其中 frag_batch.mol_idx 标记每个frag属于哪个分子

        返回: [n_frags, d_frag]
        """
        return self.gin(frag_batch.x, frag_batch.edge_index, frag_batch.batch)

    def forward(self, frag_batch: Batch, mol_idx: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        frag_batch: PyG Batch（所有分子的所有Fragment合并）
        mol_idx:    [n_frags] 每个frag属于哪个分子（0-indexed）

        返回:
          preds:    [n_molecules, n_tasks]
          lb_loss:  scalar 负载均衡损失
        """
        n_mols = int(mol_idx.max().item()) + 1

        # 1. GIN编码所有Fragment
        frag_emb = self.encode_fragments(frag_batch)  # [n_frags, d_frag]

        # 2. Router → 权重 + 负载
        weights, load = self.router(frag_emb)         # [n_frags, n_experts], [n_experts]

        # 3. 每个Expert处理所有Fragment
        expert_outs = torch.stack(
            [expert(frag_emb) for expert in self.experts], dim=1
        )  # [n_frags, n_experts, d_frag]

        # 4. 加权聚合：每个Fragment的输出 = ΣE w_E * expert_E(h)
        frag_moe_out = (weights.unsqueeze(-1) * expert_outs).sum(dim=1)
        # [n_frags, d_frag]

        # 5. Fragment → 分子级别（按分子分组求和）
        mol_emb = torch.zeros(n_mols, frag_moe_out.shape[-1],
                              device=frag_moe_out.device)
        mol_emb.scatter_add_(0,
            mol_idx.unsqueeze(-1).expand_as(frag_moe_out),
            frag_moe_out
        )  # [n_mols, d_frag]

        # 归一化（每个分子Fragment数可能不同）
        frag_count = torch.zeros(n_mols, device=frag_moe_out.device)
        frag_count.scatter_add_(0, mol_idx,
                                torch.ones(len(mol_idx), device=frag_moe_out.device))
        mol_emb = mol_emb / frag_count.unsqueeze(-1).clamp(min=1)

        # 6. 多任务预测
        preds = torch.cat([head(mol_emb) for head in self.heads], dim=-1)
        # [n_mols, n_tasks]

        # 7. 负载均衡损失（鼓励各Expert均匀使用）
        # L_lb = n_experts * sum(f_e * P_e) 其中 f_e=fraction routing to e, P_e=avg gate prob
        n_experts = len(self.experts)
        lb_loss = n_experts * (load * load).sum()

        return preds, lb_loss


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent.parent))
    from src.mk_ensemble.fragment import decompose_molecule
    from torch_geometric.data import Batch

    # 测试：2个分子，每个分解为若干Fragment
    smiles_list = [
        "C[C@@H]1CC[C@@]2([C@H]([C@H]3[C@@H](O2)C[C@@H]4[C@@]3(CC[C@H]5[C@H]4CC=C6[C@@]5(CC[C@@H](C6)O)C)C)C)OC1",
        "O[C@@H]1[C@@H](O)[C@H](O[C@@H]2CC[C@@]3(CC[C@@H]4[C@@H]3CC[C@@H]5[C@@]4(CCC(=C5)C)C)OC2)O[C@@H](CO)[C@@H]1O"
    ]

    all_frags = []
    mol_idx   = []
    for mol_i, smi in enumerate(smiles_list):
        frags = decompose_molecule(smi)
        all_frags.extend(frags)
        mol_idx.extend([mol_i] * len(frags))

    frag_batch = Batch.from_data_list(all_frags)
    mol_idx_t  = torch.tensor(mol_idx, dtype=torch.long)

    model = MKEnsemble(n_tasks=3, d_frag=128, n_experts=8)
    model.eval()
    with torch.no_grad():
        preds, lb_loss = model(frag_batch, mol_idx_t)

    print(f"Input: {len(smiles_list)} molecules, {len(all_frags)} fragments")
    print(f"Output preds shape: {preds.shape}")  # [2, 3]
    print(f"preds: {preds}")
    print(f"lb_loss: {lb_loss.item():.4f}")
    print("MKEnsemble model test PASSED ✓")
