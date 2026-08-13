"""
Stacking Ensemble: MKEnsemble + RF + XGBoost + GIN
预期性能提升: R² = 0.75-0.80
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import xgboost as xgb
import torch
from pathlib import Path
from typing import Dict, List, Tuple
import joblib

class StackingEnsemble:
    """
    Stacking集成学习框架
    Level 1: MKEnsemble, RandomForest, XGBoost, GIN
    Level 2: Ridge回归作为meta-learner
    """
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.level1_models = {}
        self.meta_model = Ridge(alpha=1.0)
        self.is_fitted = False
        
    def add_level1_model(self, name: str, model):
        """添加Level 1基础模型"""
        self.level1_models[name] = model
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: np.ndarray = None, y_val: np.ndarray = None):
        """
        训练Stacking集成模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征（可选）
            y_val: 验证标签（可选）
        """
        print("Training Level 1 models...")
        
        # 存储Level 1预测
        level1_train_preds = np.zeros((len(X_train), len(self.level1_models)))
        
        # 训练每个Level 1模型
        for i, (name, model) in enumerate(self.level1_models.items()):
            print(f"  Training {name}...")
            
            if hasattr(model, 'fit'):
                # Sklearn-style models
                model.fit(X_train, y_train)
                level1_train_preds[:, i] = model.predict(X_train)
            else:
                # Custom models (e.g., PyTorch)
                # 假设模型已经训练好
                level1_train_preds[:, i] = model.predict(X_train)
            
            print(f"    {name} R² on train: {r2_score(y_train, level1_train_preds[:, i]):.4f}")
        
        # 训练meta-learner
        print("\nTraining meta-learner (Ridge)...")
        self.meta_model.fit(level1_train_preds, y_train)
        
        # 验证集评估
        if X_val is not None and y_val is not None:
            val_preds = self.predict(X_val)
            val_r2 = r2_score(y_val, val_preds)
            print(f"Validation R²: {val_r2:.4f}")
        
        self.is_fitted = True
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        使用Stacking集成进行预测
        
        Args:
            X: 输入特征
        Returns:
            predictions: 预测值
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        # Level 1预测
        level1_preds = np.zeros((len(X), len(self.level1_models)))
        
        for i, (name, model) in enumerate(self.level1_models.items()):
            if hasattr(model, 'predict'):
                level1_preds[:, i] = model.predict(X)
            else:
                level1_preds[:, i] = model.predict(X)
        
        # Meta-learner预测
        final_preds = self.meta_model.predict(level1_preds)
        
        return final_preds
    
    def get_model_weights(self) -> Dict[str, float]:
        """获取meta-learner学到的模型权重"""
        weights = {}
        for i, name in enumerate(self.level1_models.keys()):
            weights[name] = self.meta_model.coef_[i]
        return weights
    
    def save(self, path: str):
        """保存模型"""
        save_dict = {
            'level1_models': self.level1_models,
            'meta_model': self.meta_model,
            'is_fitted': self.is_fitted
        }
        joblib.dump(save_dict, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """加载模型"""
        save_dict = joblib.load(path)
        self.level1_models = save_dict['level1_models']
        self.meta_model = save_dict['meta_model']
        self.is_fitted = save_dict['is_fitted']
        print(f"Model loaded from {path}")


def create_baseline_models(random_state: int = 42) -> Dict:
    """
    创建基线模型
    
    Returns:
        models: 包含RF和XGBoost的字典
    """
    models = {
        'RandomForest': RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=random_state,
            n_jobs=-1
        ),
        'XGBoost': xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=random_state,
            n_jobs=-1
        )
    }
    return models


def loocv_evaluation(model, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    Leave-One-Out交叉验证评估
    
    Args:
        model: 模型实例
        X: 特征
        y: 标签
    Returns:
        metrics: 包含R², RMSE, MAE, Pearson r的字典
    """
    loo = LeaveOneOut()
    predictions = np.zeros(len(y))
    
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 训练
        if hasattr(model, 'fit'):
            model.fit(X_train, y_train)
        
        # 预测
        predictions[test_idx] = model.predict(X_test)
    
    # 计算指标
    r2 = r2_score(y, predictions)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    mae = mean_absolute_error(y, predictions)
    pearson_r, _ = pearsonr(y, predictions)
    
    metrics = {
        'R2': r2,
        'RMSE': rmse,
        'MAE': mae,
        'Pearson_r': pearson_r
    }
    
    return metrics, predictions


def run_ensemble_experiment(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    assay_name: str,
    output_dir: Path
):
    """
    运行完整的集成学习实验
    
    Args:
        X: 特征矩阵
        y: 标签
        feature_names: 特征名称
        assay_name: Assay名称（DPPH/ABTS/FRAP）
        output_dir: 输出目录
    """
    print(f"\n{'='*60}")
    print(f"Running Ensemble Experiment for {assay_name}")
    print(f"{'='*60}\n")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建基线模型
    baseline_models = create_baseline_models()
    
    # 评估每个基线模型（LOOCV）
    results = []
    all_predictions = {}
    
    for model_name, model in baseline_models.items():
        print(f"\nEvaluating {model_name} with LOOCV...")
        metrics, preds = loocv_evaluation(model, X, y)
        
        print(f"  R²: {metrics['R2']:.4f}")
        print(f"  RMSE: {metrics['RMSE']:.4f}")
        print(f"  MAE: {metrics['MAE']:.4f}")
        print(f"  Pearson r: {metrics['Pearson_r']:.4f}")
        
        results.append({
            'model': model_name,
            'assay': assay_name,
            'cv_type': 'LOOCV',
            **metrics
        })
        
        all_predictions[model_name] = preds
    
    # 创建Stacking集成
    print(f"\n{'='*60}")
    print("Creating Stacking Ensemble...")
    print(f"{'='*60}\n")
    
    ensemble = StackingEnsemble()
    
    # 添加基线模型到ensemble
    for model_name, model in baseline_models.items():
        # 在全数据上训练
        model.fit(X, y)
        ensemble.add_level1_model(model_name, model)
    
    # TODO: 添加MKEnsemble和HybridMKEnsemble（需要先训练）
    # ensemble.add_level1_model('MKEnsemble', mkensemble_model)
    # ensemble.add_level1_model('HybridMKEnsemble', hybrid_model)
    
    # 评估Stacking集成（LOOCV）
    print("\nEvaluating Stacking Ensemble with LOOCV...")
    
    # 手动实现LOOCV for ensemble
    loo = LeaveOneOut()
    ensemble_predictions = np.zeros(len(y))
    
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 创建临时ensemble
        temp_ensemble = StackingEnsemble()
        
        # 训练Level 1模型
        for model_name, model_class in [
            ('RandomForest', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)),
            ('XGBoost', xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42))
        ]:
            model_class.fit(X_train, y_train)
            temp_ensemble.add_level1_model(model_name, model_class)
        
        # 训练meta-learner
        temp_ensemble.fit(X_train, y_train)
        
        # 预测
        ensemble_predictions[test_idx] = temp_ensemble.predict(X_test)
    
    # 计算ensemble指标
    ensemble_metrics = {
        'R2': r2_score(y, ensemble_predictions),
        'RMSE': np.sqrt(mean_squared_error(y, ensemble_predictions)),
        'MAE': mean_absolute_error(y, ensemble_predictions),
        'Pearson_r': pearsonr(y, ensemble_predictions)[0]
    }
    
    print(f"  R²: {ensemble_metrics['R2']:.4f}")
    print(f"  RMSE: {ensemble_metrics['RMSE']:.4f}")
    print(f"  MAE: {ensemble_metrics['MAE']:.4f}")
    print(f"  Pearson r: {ensemble_metrics['Pearson_r']:.4f}")
    
    results.append({
        'model': 'StackingEnsemble',
        'assay': assay_name,
        'cv_type': 'LOOCV',
        **ensemble_metrics
    })
    
    all_predictions['StackingEnsemble'] = ensemble_predictions
    
    # 保存结果
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / f'{assay_name}_ensemble_results.csv', index=False)
    
    # 保存预测值
    preds_df = pd.DataFrame(all_predictions)
    preds_df['true_value'] = y
    preds_df.to_csv(output_dir / f'{assay_name}_predictions.csv', index=False)
    
    # 训练最终ensemble并保存
    final_ensemble = StackingEnsemble()
    for model_name, model in baseline_models.items():
        model.fit(X, y)
        final_ensemble.add_level1_model(model_name, model)
    final_ensemble.fit(X, y)
    final_ensemble.save(output_dir / f'{assay_name}_ensemble_model.pkl')
    
    # 打印模型权重
    weights = final_ensemble.get_model_weights()
    print(f"\nMeta-learner weights:")
    for model_name, weight in weights.items():
        print(f"  {model_name}: {weight:.4f}")
    
    print(f"\nResults saved to {output_dir}")
    
    return results_df, all_predictions


if __name__ == '__main__':
    # 测试代码
    print("Ensemble framework created successfully!")
    print("\nTo use:")
    print("1. Load your data (X, y)")
    print("2. Call run_ensemble_experiment(X, y, feature_names, assay_name, output_dir)")
    print("3. Results will be saved automatically")
