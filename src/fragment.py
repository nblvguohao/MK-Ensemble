"""
src/mk_ensemble/fragment.py
Fragment分解模块：将分子SMILES分解为Murcko骨架 + BRICS侧链碎片

策略：
1. 提取Murcko骨架作为"核心Fragment"
2. 用BRICS分解得到侧链Fragment
3. 合并去重，保留≥3个重原子的Fragment
4. 每个Fragment转为 PyG Data 对象（原子特征 + 键特征）

原子特征（9维）:
  atomic_num, degree, formal_charge, num_hs, hybridization(3), is_aromatic, is_ring

键特征（3维）:
  bond_type(single/double/triple/aromatic)
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple

from rdkit import Chem
from rdkit.Chem import BRICS, AllChem, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

import torch
from torch_geometric.data import Data

# ─── 特征维度常量（供model.py引用）──────────────────────────────────────────────
ATOM_FEAT_DIM = 9
BOND_FEAT_DIM = 4

# ─── 原子/键特征提取 ──────────────────────────────────────────────────────────

HYBRIDIZATION_MAP = {
    Chem.rdchem.HybridizationType.SP:  0,
    Chem.rdchem.HybridizationType.SP2: 1,
    Chem.rdchem.HybridizationType.SP3: 2,
}

BOND_TYPE_MAP = {
    Chem.rdchem.BondType.SINGLE:    0,
    Chem.rdchem.BondType.DOUBLE:    1,
    Chem.rdchem.BondType.TRIPLE:    2,
    Chem.rdchem.BondType.AROMATIC:  3,
}

def atom_features(atom: Chem.rdchem.Atom) -> List[float]:
    hyb = HYBRIDIZATION_MAP.get(atom.GetHybridization(), 0)
    hyb_vec = [0, 0, 0]
    hyb_vec[hyb] = 1
    return [
        atom.GetAtomicNum() / 100.0,          # 归一化原子序数
        atom.GetDegree() / 6.0,               # 度数
        float(atom.GetFormalCharge()),         # 形式电荷
        atom.GetTotalNumHs() / 4.0,           # 氢原子数
        *hyb_vec,                             # 杂化方式 one-hot (3维)
        float(atom.GetIsAromatic()),           # 是否芳香
        float(atom.IsInRing()),               # 是否在环中
    ]  # 共 1+1+1+1+3+1+1 = 9 维

def bond_features(bond: Chem.rdchem.Bond) -> List[float]:
    bt = BOND_TYPE_MAP.get(bond.GetBondType(), 0)
    feat = [0.0, 0.0, 0.0, 0.0]
    feat[bt] = 1.0
    return feat  # 4维 one-hot

def mol_to_pyg(mol: Chem.Mol) -> Data | None:
    """将RDKit Mol对象转换为PyG Data"""
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    # 原子特征矩阵
    x = torch.tensor(
        [atom_features(a) for a in mol.GetAtoms()],
        dtype=torch.float
    )  # [n_atoms, ATOM_FEAT_DIM]

    # 边列表和边特征（无向图，每条边加双向）
    edge_index = []
    edge_attr  = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = bond_features(bond)
        edge_index += [[i, j], [j, i]]
        edge_attr  += [feat, feat]

    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr  = torch.tensor(edge_attr,  dtype=torch.float)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr  = torch.zeros((0, BOND_FEAT_DIM), dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                num_nodes=mol.GetNumAtoms())

# ─── Fragment分解 ──────────────────────────────────────────────────────────────

def decompose_molecule(smiles: str, min_heavy_atoms: int = 3) -> List[Data]:
    """
    将SMILES分解为Fragment列表（PyG Data对象）

    返回：[fragment_graph, ...]
    至少包含整个分子本身（fallback）
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    frags_smi = set()

    # 1. 整个分子作为一个Fragment（全局视图）
    frags_smi.add(Chem.MolToSmiles(mol))

    # 2. Murcko骨架
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        if scaf and scaf.GetNumAtoms() >= min_heavy_atoms:
            frags_smi.add(Chem.MolToSmiles(scaf))
    except Exception:
        pass

    # 3. BRICS分解
    try:
        brics_frags = BRICS.BRICSDecompose(mol)
        for smi in brics_frags:
            # BRICS会在切割点添加"[n*]"标记，需要清理
            clean = Chem.MolFromSmiles(smi)
            if clean and clean.GetNumAtoms() >= min_heavy_atoms:
                frags_smi.add(Chem.MolToSmiles(clean))
    except Exception:
        pass

    # 4. 转为PyG Data
    result = []
    for smi in frags_smi:
        frag_mol = Chem.MolFromSmiles(smi)
        if frag_mol and frag_mol.GetNumAtoms() >= min_heavy_atoms:
            g = mol_to_pyg(frag_mol)
            if g is not None:
                result.append(g)

    # 确保至少有整个分子
    if not result:
        g = mol_to_pyg(mol)
        if g is not None:
            result.append(g)

    return result

def smiles_to_fragments(smiles_list: List[str]) -> List[List[Data]]:
    """批量处理SMILES列表，返回每个分子的Fragment列表"""
    return [decompose_molecule(smi) for smi in smiles_list]


if __name__ == "__main__":
    # 快速测试
    test_smi = "O[C@@H]1[C@@H](O)[C@H](O[C@@H]2CC[C@@]3(CC[C@@H]4[C@H]3CC[C@@H]5[C@@]4(CCC(=C5)C)C)OC2)O[C@@H](CO)[C@@H]1O"
    frags = decompose_molecule(test_smi)
    print(f"SMILES → {len(frags)} fragments")
    for i, f in enumerate(frags):
        print(f"  Fragment {i}: {f.num_nodes} atoms, {f.edge_index.shape[1]//2} bonds, x.shape={f.x.shape}")
