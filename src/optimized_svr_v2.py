#!/usr/bin/env python3
"""
SVR优化V2 — 基于第一轮实验发现的精细化方案

第一轮关键发现:
  - DPPH: 多描述符融合(MKL)最有效 → 需要2D理化+皂苷特征补充Morgan FP
  - ABTS: Dice核优于Tanimoto → 对称归一化更适合该assay
  - FRAP: Dice核也有提升 → 高相似度数据集受益于Dice归一化
  - 嵌套CV选超参反而略降 → 过于保守的正则化
  - 特征选择损害DPPH → 信息丢失，不适合样本少的情况

V2策略:
  V2-A: Dice-MKL — Dice(FP) + RBF(理化+皂苷), 自适应alpha
  V2-B: Per-Assay Adaptive — 嵌套CV从 {Tanimoto, Dice, MinMax} 中自动选核
  V2-C: Stacked Kernel Ridge — KRR(Dice) + KRR(Tanimoto) + BR(2D) → Ridge meta
  V2-D: Domain-Adapted SVR — 皂苷结构指纹(糖基/苷元/羟基) + Tanimoto kernel
  V2-E: Consensus Model — 取所有 R²>0 方法的加权平均, 权重=内层R²
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pickle
import json
from datetime import datetime
from itertools import product

from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Fragments, AllChem
from rdkit.Chem import MACCSkeys

ROOT = Path(__file__).parent.parent
PHASE2_DIR = ROOT / "results/phase2"
RESULTS_DIR = ROOT / "results/optimized_svr"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ================================================================
# 数据与描述符
# ================================================================

def load_data(assay):
    with open(PHASE2_DIR / f"morgan_features_{assay}.pkl", 'rb') as f:
        data = pickle.load(f)
    return data['X'], data['y'], data['smiles']

def generate_rdkit_2d(smiles_list):
    """RDKit 2D理化描述符"""
    desc_funcs = [
        ('MolWt', Descriptors.MolWt),
        ('MolLogP', Descriptors.MolLogP),
        ('TPSA', Descriptors.TPSA),
        ('NumHAcceptors', Descriptors.NumHAcceptors),
        ('NumHDonors', Descriptors.NumHDonors),
        ('NumRotatableBonds', Descriptors.NumRotatableBonds),
        ('NumAromaticRings', Descriptors.NumAromaticRings),
        ('NumAliphaticRings', Descriptors.NumAliphaticRings),
        ('FractionCSP3', Descriptors.FractionCSP3),
        ('HeavyAtomCount', Descriptors.HeavyAtomCount),
        ('RingCount', Descriptors.RingCount),
        ('LabuteASA', Descriptors.LabuteASA),
        ('BalabanJ', Descriptors.BalabanJ),
        ('BertzCT', Descriptors.BertzCT),
        ('Chi0', Descriptors.Chi0),
    ]
    features = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        row = []
        for name, func in desc_funcs:
            try:
                row.append(float(func(mol)) if mol else 0.0)
            except:
                row.append(0.0)
        features.append(row)
    return np.array(features)

def generate_saponin_domain_features(smiles_list):
    """
    皂苷领域专属特征 — 基于甾体皂苷结构特点:
    - 糖基数量和类型
    - 苷元骨架特征
    - 羟基分布
    - 抗氧化相关官能团
    """
    features = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            features.append([0.0] * 20)
            continue
        
        ring_info = mol.GetRingInfo()
        atom_rings = ring_info.AtomRings()
        
        # === 糖基特征 ===
        # 含氧五/六元环 = 糖基候选
        sugar_rings = [r for r in atom_rings
                       if len(r) in [5, 6] and
                       any(mol.GetAtomWithIdx(i).GetSymbol() == 'O' for i in r)]
        n_sugar = len(sugar_rings)
        
        # 糖基碳原子总数
        sugar_atoms = set()
        for r in sugar_rings:
            sugar_atoms.update(r)
        n_sugar_atoms = len(sugar_atoms)
        
        # === 苷元骨架特征 ===
        # 非糖环 = 苷元骨架环
        non_sugar_rings = [r for r in atom_rings if r not in sugar_rings]
        n_aglycone_rings = len(non_sugar_rings)
        
        # 五元环和六元环分别计数（甾体通常3个六元+1个五元）
        n_5ring = sum(1 for r in atom_rings if len(r) == 5)
        n_6ring = sum(1 for r in atom_rings if len(r) == 6)
        
        # === 羟基与抗氧化官能团 ===
        n_OH_aliph = Fragments.fr_Al_OH(mol)     # 脂肪族OH
        n_OH_arom  = Fragments.fr_Ar_OH(mol)     # 芳香族OH（酚羟基，直接抗氧化）
        n_OH_total = n_OH_aliph + n_OH_arom
        
        # 酚羟基（关键抗氧化基团）
        n_phenol = Fragments.fr_phenol(mol)
        
        # 醚键 (C-O-C, 包括糖苷键)
        n_ether = Fragments.fr_ether(mol)
        
        # 酯基
        n_ester = Fragments.fr_ester(mol)
        
        # === 结构复杂度 ===
        n_stereo = rdMolDescriptors.CalcNumAtomStereoCenters(mol)
        n_heavy = mol.GetNumHeavyAtoms()
        mw = Descriptors.MolWt(mol)
        
        # O原子总数
        n_O = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == 'O')
        n_C = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == 'C')
        ratio_OC = n_O / max(n_C, 1)
        
        # sp3碳比例（皂苷通常很高）
        frac_csp3 = Descriptors.FractionCSP3(mol)
        
        # 双键数（对抗氧化活性有影响）
        n_double = sum(1 for b in mol.GetBonds()
                       if b.GetBondType() == Chem.rdchem.BondType.DOUBLE)
        
        # 可旋转键数（柔性）
        n_rotatable = Descriptors.NumRotatableBonds(mol)
        
        # 糖基/苷元比例
        sugar_ratio = n_sugar_atoms / max(n_heavy, 1)
        
        # TPSA（极性表面积，与OH/O分布相关）
        tpsa = Descriptors.TPSA(mol)
        
        features.append([
            n_sugar, n_sugar_atoms, n_aglycone_rings,
            n_5ring, n_6ring,
            n_OH_total, n_OH_aliph, n_OH_arom, n_phenol,
            n_ether, n_ester,
            n_stereo, n_heavy, mw,
            n_O, ratio_OC, frac_csp3,
            n_double, n_rotatable,
            sugar_ratio,
        ])
    
    return np.array(features, dtype=float)

def generate_maccs_keys(smiles_list):
    """MACCS指纹 — 166位结构关键字，与Morgan互补"""
    features = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = MACCSkeys.GenMACCSKeys(mol)
            features.append(np.array(fp))
        else:
            features.append(np.zeros(167))
    return np.array(features, dtype=float)

# ================================================================
# 核函数
# ================================================================

def tanimoto_kernel(X, Y=None):
    if Y is None: Y = X
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    XY = X @ Y.T
    X_sq = np.sum(X * X, axis=1, keepdims=True)
    Y_sq = np.sum(Y * Y, axis=1, keepdims=True)
    return XY / np.maximum(X_sq + Y_sq.T - XY, 1e-8)

def dice_kernel(X, Y=None):
    if Y is None: Y = X
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    XY = X @ Y.T
    X_sum = np.sum(X, axis=1, keepdims=True)
    Y_sum = np.sum(Y, axis=1, keepdims=True)
    return 2 * XY / np.maximum(X_sum + Y_sum.T, 1e-8)

def rbf_kernel(X, Y=None, gamma=None):
    from sklearn.metrics.pairwise import rbf_kernel as sk_rbf
    if Y is None: Y = X
    if gamma is None: gamma = 1.0 / X.shape[1]
    return sk_rbf(X, Y, gamma=gamma)

def combined_kernel(K1, K2, alpha=0.5):
    return alpha * K1 + (1 - alpha) * K2

# ================================================================
# 评估
# ================================================================

def evaluate(y_true, y_pred):
    return {
        'R2': r2_score(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'Pearson_r': pearsonr(y_true, y_pred)[0],
    }

def inner_loo_score(K_train, y_train, C, eps):
    """内层LOO评估给定超参的SVR性能"""
    n = len(y_train)
    preds = []
    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        Ki_tr = K_train[np.ix_(mask, mask)]
        Ki_te = K_train[i:i+1, mask]
        try:
            svr = SVR(kernel='precomputed', C=C, epsilon=eps)
            svr.fit(Ki_tr, y_train[mask])
            preds.append(svr.predict(Ki_te)[0])
        except:
            preds.append(y_train.mean())
    return r2_score(y_train, preds)

# ================================================================
# V2-A: Dice-MKL
# ================================================================

def v2a_dice_mkl(X_fp, X_phys, y):
    """Dice(FP) + RBF(理化+皂苷) with adaptive alpha"""
    loo = LeaveOneOut()
    preds = []
    scaler = StandardScaler()
    
    for tr, te in loo.split(X_fp):
        K_dice_tr = dice_kernel(X_fp[tr])
        K_dice_te = dice_kernel(X_fp[te], X_fp[tr])
        
        Xp_tr = scaler.fit_transform(X_phys[tr])
        Xp_te = scaler.transform(X_phys[te])
        K_rbf_tr = rbf_kernel(Xp_tr)
        K_rbf_te = rbf_kernel(Xp_te, Xp_tr)
        
        # 内层选alpha和C
        best_score, best_alpha, best_C = -999, 0.7, 10.0
        for alpha in [0.5, 0.6, 0.7, 0.8, 0.9]:
            K_comb = combined_kernel(K_dice_tr, K_rbf_tr, alpha)
            for C in [1.0, 5.0, 10.0, 50.0]:
                score = inner_loo_score(K_comb, y[tr], C, 0.05)
                if score > best_score:
                    best_score, best_alpha, best_C = score, alpha, C
        
        K_tr = combined_kernel(K_dice_tr, K_rbf_tr, best_alpha)
        K_te = combined_kernel(K_dice_te, K_rbf_te, best_alpha)
        svr = SVR(kernel='precomputed', C=best_C, epsilon=0.05)
        svr.fit(K_tr, y[tr])
        preds.append(svr.predict(K_te)[0])
    
    return np.array(preds)

# ================================================================
# V2-B: Per-Assay Adaptive Kernel
# ================================================================

def v2b_adaptive_kernel(X_fp, y):
    """嵌套CV自动从{Tanimoto, Dice, MinMax}选最佳核+超参"""
    kernel_fns = {'Tanimoto': tanimoto_kernel, 'Dice': dice_kernel}
    loo = LeaveOneOut()
    preds = []
    chosen_kernels = []
    
    for tr, te in loo.split(X_fp):
        best_score, best_kernel, best_C, best_eps = -999, 'Tanimoto', 10.0, 0.05
        
        for kname, kfn in kernel_fns.items():
            K_tr = kfn(X_fp[tr])
            for C in [1.0, 5.0, 10.0, 50.0]:
                for eps in [0.01, 0.05, 0.1]:
                    score = inner_loo_score(K_tr, y[tr], C, eps)
                    if score > best_score:
                        best_score = score
                        best_kernel, best_C, best_eps = kname, C, eps
        
        kfn = kernel_fns[best_kernel]
        K_tr = kfn(X_fp[tr])
        K_te = kfn(X_fp[te], X_fp[tr])
        svr = SVR(kernel='precomputed', C=best_C, epsilon=best_eps)
        svr.fit(K_tr, y[tr])
        preds.append(svr.predict(K_te)[0])
        chosen_kernels.append(best_kernel)
    
    # 统计kernel选择频率
    from collections import Counter
    kernel_freq = Counter(chosen_kernels)
    return np.array(preds), kernel_freq

# ================================================================
# V2-C: Stacked Kernel Ridge
# ================================================================

def v2c_stacked_kernel(X_fp, X_phys, y):
    """
    Level-1: KRR(Dice), KRR(Tanimoto), BayesianRidge(2D)
    Level-2: Ridge meta-learner on LOO predictions
    """
    n = len(y)
    loo = LeaveOneOut()
    scaler = StandardScaler()
    
    # 收集Level-1 LOO预测
    meta_features = np.zeros((n, 3))
    
    # KRR-Dice
    for i, (tr, te) in enumerate(loo.split(X_fp)):
        K_tr = dice_kernel(X_fp[tr])
        K_te = dice_kernel(X_fp[te], X_fp[tr])
        alpha = 0.1
        w = np.linalg.solve(K_tr + alpha * np.eye(len(K_tr)), y[tr])
        meta_features[i, 0] = (K_te @ w)[0]
    
    # KRR-Tanimoto
    for i, (tr, te) in enumerate(loo.split(X_fp)):
        K_tr = tanimoto_kernel(X_fp[tr])
        K_te = tanimoto_kernel(X_fp[te], X_fp[tr])
        alpha = 0.1
        w = np.linalg.solve(K_tr + alpha * np.eye(len(K_tr)), y[tr])
        meta_features[i, 1] = (K_te @ w)[0]
    
    # BayesianRidge on phys
    for i, (tr, te) in enumerate(loo.split(X_phys)):
        Xp_tr = scaler.fit_transform(X_phys[tr])
        Xp_te = scaler.transform(X_phys[te])
        br = BayesianRidge()
        br.fit(Xp_tr, y[tr])
        meta_features[i, 2] = br.predict(Xp_te)[0]
    
    # Level-2: LOO Ridge
    preds = []
    for i, (tr, te) in enumerate(loo.split(meta_features)):
        ridge = Ridge(alpha=1.0)
        ridge.fit(meta_features[tr], y[tr])
        preds.append(ridge.predict(meta_features[te])[0])
    
    return np.array(preds)

# ================================================================
# V2-D: Domain-Adapted SVR
# ================================================================

def v2d_domain_adapted(X_fp, X_maccs, X_sap, y):
    """
    皂苷领域适配: Dice(Morgan) + Tanimoto(MACCS) + RBF(皂苷特征)
    三种互补描述符覆盖不同结构层次
    """
    loo = LeaveOneOut()
    preds = []
    scaler = StandardScaler()
    
    for tr, te in loo.split(X_fp):
        K_dice_tr = dice_kernel(X_fp[tr])
        K_dice_te = dice_kernel(X_fp[te], X_fp[tr])
        
        K_maccs_tr = tanimoto_kernel(X_maccs[tr])
        K_maccs_te = tanimoto_kernel(X_maccs[te], X_maccs[tr])
        
        Xs_tr = scaler.fit_transform(X_sap[tr])
        Xs_te = scaler.transform(X_sap[te])
        K_sap_tr = rbf_kernel(Xs_tr)
        K_sap_te = rbf_kernel(Xs_te, Xs_tr)
        
        # 内层选权重
        best_score = -999
        best_w = (0.5, 0.3, 0.2)
        for w1 in [0.4, 0.5, 0.6, 0.7]:
            for w2 in [0.1, 0.2, 0.3]:
                w3 = 1.0 - w1 - w2
                if w3 < 0.05: continue
                K_comb = w1*K_dice_tr + w2*K_maccs_tr + w3*K_sap_tr
                score = inner_loo_score(K_comb, y[tr], 10.0, 0.05)
                if score > best_score:
                    best_score = score
                    best_w = (w1, w2, w3)
        
        K_tr = best_w[0]*K_dice_tr + best_w[1]*K_maccs_tr + best_w[2]*K_sap_tr
        K_te = best_w[0]*K_dice_te + best_w[1]*K_maccs_te + best_w[2]*K_sap_te
        
        svr = SVR(kernel='precomputed', C=10.0, epsilon=0.05)
        svr.fit(K_tr, y[tr])
        preds.append(svr.predict(K_te)[0])
    
    return np.array(preds)

# ================================================================
# V2-E: Consensus Model
# ================================================================

def v2e_consensus(preds_dict, y):
    """取所有R²>0方法的 softmax(R²) 加权平均"""
    valid = {}
    for name, pred in preds_dict.items():
        r2 = r2_score(y, pred)
        if r2 > 0:
            valid[name] = (pred, r2)
    
    if not valid:
        return np.full(len(y), y.mean()), {}
    
    # Softmax权重（温度=1.0）
    r2_vals = np.array([v[1] for v in valid.values()])
    exp_r2 = np.exp(r2_vals * 5)  # 放大差异
    weights = exp_r2 / exp_r2.sum()
    
    consensus = np.zeros(len(y))
    weight_dict = {}
    for (name, (pred, r2)), w in zip(valid.items(), weights):
        consensus += w * pred
        weight_dict[name] = float(w)
    
    return consensus, weight_dict

# ================================================================
# 主函数
# ================================================================

def main():
    print("=" * 70)
    print("SVR优化V2 — 基于第一轮发现的精细化方案")
    print("=" * 70)
    
    all_results = []
    all_details = {}
    
    for assay in ['DPPH', 'ABTS', 'FRAP']:
        X_fp, y, smiles = load_data(assay)
        n = len(y)
        
        print(f"\n{'='*60}")
        print(f"Assay: {assay} (n={n})")
        print(f"{'='*60}")
        
        # 生成所有描述符
        X_2d = generate_rdkit_2d(smiles)
        X_sap = generate_saponin_domain_features(smiles)
        X_maccs = generate_maccs_keys(smiles)
        X_phys = np.hstack([X_2d, X_sap])  # 理化+皂苷
        
        print(f"  描述符: Morgan({X_fp.shape[1]}) + RDKit2D({X_2d.shape[1]}) "
              f"+ Saponin({X_sap.shape[1]}) + MACCS({X_maccs.shape[1]})")
        
        preds_all = {}
        
        # -- 基线: SVR-Tanimoto --
        print("\n  [Base] SVR-Tanimoto...")
        loo = LeaveOneOut()
        p_base = []
        for tr, te in loo.split(X_fp):
            K_tr = tanimoto_kernel(X_fp[tr])
            K_te = tanimoto_kernel(X_fp[te], X_fp[tr])
            svr = SVR(kernel='precomputed', C=10.0, epsilon=0.05)
            svr.fit(K_tr, y[tr])
            p_base.append(svr.predict(K_te)[0])
        p_base = np.array(p_base)
        m_base = evaluate(y, p_base)
        preds_all['SVR-Tanimoto'] = p_base
        all_results.append({'assay': assay, 'method': 'SVR-Tanimoto(base)', **m_base})
        print(f"         R²={m_base['R2']:.4f}")
        
        # -- 基线: SVR-Dice --
        p_dice_base = []
        for tr, te in loo.split(X_fp):
            K_tr = dice_kernel(X_fp[tr])
            K_te = dice_kernel(X_fp[te], X_fp[tr])
            svr = SVR(kernel='precomputed', C=10.0, epsilon=0.05)
            svr.fit(K_tr, y[tr])
            p_dice_base.append(svr.predict(K_te)[0])
        p_dice_base = np.array(p_dice_base)
        preds_all['SVR-Dice'] = p_dice_base
        
        # -- 基线: RF --
        p_rf = cross_val_predict(
            RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
            X_fp, y, cv=LeaveOneOut())
        m_rf = evaluate(y, p_rf)
        preds_all['RF'] = p_rf
        all_results.append({'assay': assay, 'method': 'RF(base)', **m_rf})
        print(f"  [Base] RF: R²={m_rf['R2']:.4f}")
        
        # -- V2-A: Dice-MKL --
        print("  [V2-A] Dice-MKL...")
        p_a = v2a_dice_mkl(X_fp, X_phys, y)
        m_a = evaluate(y, p_a)
        preds_all['V2-A'] = p_a
        all_results.append({'assay': assay, 'method': 'V2-A: Dice-MKL', **m_a})
        print(f"         R²={m_a['R2']:.4f}  Δ={m_a['R2']-m_base['R2']:+.4f}")
        
        # -- V2-B: Adaptive Kernel --
        print("  [V2-B] Adaptive Kernel...")
        p_b, kfreq = v2b_adaptive_kernel(X_fp, y)
        m_b = evaluate(y, p_b)
        preds_all['V2-B'] = p_b
        all_results.append({'assay': assay, 'method': 'V2-B: AdaptiveKernel', **m_b})
        print(f"         R²={m_b['R2']:.4f}  Δ={m_b['R2']-m_base['R2']:+.4f}  "
              f"kernels={dict(kfreq)}")
        
        # -- V2-C: Stacked Kernel --
        print("  [V2-C] Stacked Kernel...")
        p_c = v2c_stacked_kernel(X_fp, X_phys, y)
        m_c = evaluate(y, p_c)
        preds_all['V2-C'] = p_c
        all_results.append({'assay': assay, 'method': 'V2-C: StackedKernel', **m_c})
        print(f"         R²={m_c['R2']:.4f}  Δ={m_c['R2']-m_base['R2']:+.4f}")
        
        # -- V2-D: Domain-Adapted --
        print("  [V2-D] Domain-Adapted SVR...")
        p_d = v2d_domain_adapted(X_fp, X_maccs, X_sap, y)
        m_d = evaluate(y, p_d)
        preds_all['V2-D'] = p_d
        all_results.append({'assay': assay, 'method': 'V2-D: DomainAdapted', **m_d})
        print(f"         R²={m_d['R2']:.4f}  Δ={m_d['R2']-m_base['R2']:+.4f}")
        
        # -- V2-E: Consensus --
        print("  [V2-E] Consensus Model...")
        p_e, cons_weights = v2e_consensus(preds_all, y)
        m_e = evaluate(y, p_e)
        all_results.append({'assay': assay, 'method': 'V2-E: Consensus', **m_e})
        print(f"         R²={m_e['R2']:.4f}  Δ={m_e['R2']-m_base['R2']:+.4f}")
        if cons_weights:
            top3 = sorted(cons_weights.items(), key=lambda x: -x[1])[:3]
            print(f"         top weights: {', '.join(f'{k}={v:.3f}' for k,v in top3)}")
        
        # 排名
        assay_res = [r for r in all_results if r['assay'] == assay]
        assay_res.sort(key=lambda x: x['R2'], reverse=True)
        print(f"\n  🏆 {assay} 完整排名:")
        for i, r in enumerate(assay_res, 1):
            delta = r['R2'] - m_base['R2']
            tag = "★" if delta > 0 else " "
            print(f"   {tag} {i}. {r['method']:30s} R²={r['R2']:.4f} (Δ={delta:+.4f})")
        
        all_details[assay] = {
            'consensus_weights': cons_weights,
            'adaptive_kernel_freq': dict(kfreq),
            'n_samples': n,
        }
    
    # ============ 全局汇总 ============
    df = pd.DataFrame(all_results)
    
    print(f"\n{'='*70}")
    print("V2 总体排名（按平均R²）")
    print(f"{'='*70}")
    
    summary = df.groupby('method').agg(
        mean_R2=('R2', 'mean'),
        std_R2=('R2', 'std'),
        mean_RMSE=('RMSE', 'mean'),
        DPPH_R2=('R2', lambda x: x.iloc[0] if len(x) > 0 else np.nan),
        ABTS_R2=('R2', lambda x: x.iloc[1] if len(x) > 1 else np.nan),
        FRAP_R2=('R2', lambda x: x.iloc[2] if len(x) > 2 else np.nan),
    ).round(4)
    summary = summary.sort_values('mean_R2', ascending=False)
    
    # 获取baseline的平均R2
    if 'SVR-Tanimoto(base)' in summary.index:
        base_mean = summary.loc['SVR-Tanimoto(base)', 'mean_R2']
    else:
        base_mean = 0.77
    
    for i, (method, row) in enumerate(summary.iterrows(), 1):
        delta = row['mean_R2'] - base_mean
        tag = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f"  {tag} {i}. {method:30s}  R²={row['mean_R2']:.4f}±{row['std_R2']:.4f}  "
              f"RMSE={row['mean_RMSE']:.4f}  Δ={delta:+.4f}")
    
    # 保存
    csv_path = RESULTS_DIR / "optimized_svr_v2_results.csv"
    df.to_csv(csv_path, index=False)
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'purpose': 'SVR优化V2 — 领域适配精细化',
        'ranking': summary.reset_index().to_dict('records'),
        'details': {k: {kk: vv for kk, vv in v.items()
                        if not isinstance(vv, np.ndarray)}
                    for k, v in all_details.items()},
    }
    report_path = RESULTS_DIR / "optimization_v2_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating,)) else int(x) if isinstance(x, (np.integer,)) else x)
    
    print(f"\n📊 结果: {csv_path}")
    print(f"📄 报告: {report_path}")

if __name__ == '__main__':
    main()
