#!/usr/bin/env python3
"""
训练HybridMKEnsemble模型
使用现有数据集，目标：DPPH R² > 0.65
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch_geometric.data import Data, DataLoader
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import argparse
import json
from datetime import datetime

ROOT = Path(__file__).parent.parent
DATA_ROOT = ROOT / "data/processed"
RESULTS_ROOT = ROOT / "results/phase3_hybrid"
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

def mol_to_graph(smiles):
    """将SMILES转换为图数据"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # 节点特征（原子）
    atom_features = []
    for atom in mol.GetAtoms():
        features = [
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            atom.GetHybridization().real,
            int(atom.GetIsAromatic()),
            atom.GetTotalNumHs(),
            atom.GetNumRadicalElectrons(),
            int(atom.IsInRing()),
            atom.GetMass()
        ]
        atom_features.append(features)
    
    x = torch.tensor(atom_features, dtype=torch.float)
    
    # 边索引
    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    return Data(x=x, edge_index=edge_index)

def fragment_molecule(smiles):
    """简化版片段分解 - 使用BRICS"""
    from rdkit.Chem import BRICS
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [smiles]  # 返回原分子
    
    # BRICS分解
    fragments = list(BRICS.BRICSDecompose(mol))
    
    if len(fragments) == 0:
        return [smiles]  # 如果无法分解，返回原分子
    
    return fragments[:10]  # 最多10个片段

class SimpleHybridModel(nn.Module):
    """简化版HybridMKEnsemble用于快速训练"""
    def __init__(self, input_dim=9, hidden_dim=128):
        super().__init__()
        
        # 全局分子编码器
        self.global_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 片段编码器（简化）
        self.fragment_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 32),
            nn.ReLU()
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(64 + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x_global, x_fragment):
        """
        Args:
            x_global: 全局分子特征 [batch, input_dim]
            x_fragment: 片段特征 [batch, input_dim]
        """
        global_repr = self.global_encoder(x_global)
        fragment_repr = self.fragment_encoder(x_fragment)
        
        combined = torch.cat([global_repr, fragment_repr], dim=-1)
        output = self.fusion(combined)
        
        return output

def compute_molecular_features(smiles):
    """计算分子特征（简化版）"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(9)
    
    features = [
        Chem.Descriptors.MolWt(mol),
        Chem.Descriptors.MolLogP(mol),
        Chem.Descriptors.NumHDonors(mol),
        Chem.Descriptors.NumHAcceptors(mol),
        Chem.Descriptors.TPSA(mol),
        Chem.Descriptors.NumRotatableBonds(mol),
        Chem.Descriptors.NumAromaticRings(mol),
        Chem.Descriptors.NumAliphaticRings(mol),
        mol.GetNumAtoms()
    ]
    
    return np.array(features)

def prepare_data(assay_name):
    """准备训练数据"""
    print(f"\n准备{assay_name}数据...")
    
    # 加载数据
    dataset = pd.read_csv(DATA_ROOT / "antioxidant_dataset.csv")
    assay_data = dataset[dataset['assay_type'] == assay_name].copy()
    
    print(f"样本数: {len(assay_data)}")
    
    # 计算特征
    X_global = []
    X_fragment = []
    y = []
    
    for idx, row in assay_data.iterrows():
        smiles = row['SMILES']
        
        # 全局特征
        global_feat = compute_molecular_features(smiles)
        
        # 片段特征（简化：使用第一个片段）
        fragments = fragment_molecule(smiles)
        fragment_feat = compute_molecular_features(fragments[0])
        
        X_global.append(global_feat)
        X_fragment.append(fragment_feat)
        y.append(row['pIC50'])
    
    X_global = np.array(X_global)
    X_fragment = np.array(X_fragment)
    y = np.array(y)
    
    # 标准化
    from sklearn.preprocessing import StandardScaler
    scaler_global = StandardScaler()
    scaler_fragment = StandardScaler()
    
    X_global = scaler_global.fit_transform(X_global)
    X_fragment = scaler_fragment.fit_transform(X_fragment)
    
    return X_global, X_fragment, y, scaler_global, scaler_fragment

def train_model(assay_name, epochs=200, lr=0.001, device='cuda'):
    """训练模型"""
    print(f"\n{'='*60}")
    print(f"训练HybridMKEnsemble - {assay_name}")
    print(f"{'='*60}")
    
    # 准备数据
    X_global, X_fragment, y, scaler_g, scaler_f = prepare_data(assay_name)
    
    # 转换为tensor
    X_global = torch.FloatTensor(X_global)
    X_fragment = torch.FloatTensor(X_fragment)
    y = torch.FloatTensor(y)
    
    # 创建模型
    model = SimpleHybridModel(input_dim=9, hidden_dim=128)
    
    # 检查设备
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("CUDA不可用，使用CPU训练")
    
    model = model.to(device)
    X_global = X_global.to(device)
    X_fragment = X_fragment.to(device)
    y = y.to(device)
    
    # 优化器和损失
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    # 训练
    best_loss = float('inf')
    patience = 30
    patience_counter = 0
    
    history = {
        'train_loss': [],
        'train_r2': []
    }
    
    print(f"\n开始训练...")
    print(f"设备: {device}")
    print(f"样本数: {len(y)}")
    
    for epoch in range(epochs):
        model.train()
        
        # 前向传播
        optimizer.zero_grad()
        pred = model(X_global, X_fragment).squeeze()
        loss = criterion(pred, y)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 计算R²
        with torch.no_grad():
            pred_np = pred.cpu().numpy()
            y_np = y.cpu().numpy()
            r2 = r2_score(y_np, pred_np)
        
        history['train_loss'].append(loss.item())
        history['train_r2'].append(r2)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}, R²: {r2:.4f}")
        
        # Early stopping
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
            # 保存最佳模型
            torch.save({
                'model_state_dict': model.state_dict(),
                'scaler_global': scaler_g,
                'scaler_fragment': scaler_f,
                'assay': assay_name
            }, RESULTS_ROOT / f'hybrid_mkensemble_{assay_name}.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # 加载最佳模型并评估
    checkpoint = torch.load(RESULTS_ROOT / f'hybrid_mkensemble_{assay_name}.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    with torch.no_grad():
        pred = model(X_global, X_fragment).squeeze()
        pred_np = pred.cpu().numpy()
        y_np = y.cpu().numpy()
        
        r2 = r2_score(y_np, pred_np)
        rmse = np.sqrt(mean_squared_error(y_np, pred_np))
        mae = mean_absolute_error(y_np, pred_np)
    
    results = {
        'assay': assay_name,
        'model': 'HybridMKEnsemble',
        'n_samples': len(y),
        'R2': float(r2),
        'RMSE': float(rmse),
        'MAE': float(mae),
        'best_epoch': len(history['train_loss']) - patience,
        'timestamp': datetime.now().isoformat()
    }
    
    print(f"\n{'='*60}")
    print(f"训练完成 - {assay_name}")
    print(f"{'='*60}")
    print(f"R²: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    
    # 保存结果
    with open(RESULTS_ROOT / f'results_{assay_name}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # 保存训练历史
    history_df = pd.DataFrame(history)
    history_df.to_csv(RESULTS_ROOT / f'training_history_{assay_name}.csv', index=False)
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--assay', type=str, default='DPPH', choices=['DPPH', 'ABTS', 'FRAP'])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    results = train_model(args.assay, args.epochs, args.lr, args.device)
    
    print(f"\n模型已保存至: {RESULTS_ROOT}")
    print(f"结果: {results}")

if __name__ == '__main__':
    main()
