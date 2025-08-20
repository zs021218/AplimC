#!/usr/bin/env python3
"""
基于Stokes和荧光统计量的机器学习分类器
支持多种经典机器学习算法和集成方法
"""

import numpy as np
import pandas as pd
import h5py
import json
import argparse
from pathlib import Path
from typing import Dict, Tuple, Any
import joblib
import time
from datetime import datetime

# 机器学习库
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    ExtraTreesClassifier, VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

# 评估和预处理
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_auc_score
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

# 可视化
import matplotlib.pyplot as plt
import seaborn as sns

class StatisticsClassifier:
    """基于统计量的分类器"""
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.scalers = {}
        self.models = {}
        self.feature_selector = None
        self.pca = None
        self.class_names = {
            0: 'CG', 1: 'IG', 2: 'PS3', 3: 'PS6', 4: 'PS10', 5: 'QDDB',
            6: 'QZQG', 7: 'SG', 8: 'TP', 9: 'TS', 10: 'YMXH', 11: 'YXXB'
        }
        
    def _load_config(self, config_path: str) -> Dict:
        """加载配置或使用默认配置"""
        default_config = {
            'data_path': 'data/processed/multimodal_statistics.h5',
            'output_dir': 'experiments/statistics_classification',
            'preprocessing': {
                'scaler': 'standard',  # 'standard', 'minmax', 'robust'
                'feature_selection': True,
                'n_features': 100,
                'pca': False,
                'pca_components': 50
            },
            'models': {
                'random_forest': {
                    'n_estimators': 200,
                    'max_depth': 20,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': 42
                },
                'gradient_boosting': {
                    'n_estimators': 200,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'random_state': 42
                },
                'svm': {
                    'C': 10.0,
                    'gamma': 'scale',
                    'kernel': 'rbf',
                    'random_state': 42
                },
                'logistic_regression': {
                    'C': 1.0,
                    'max_iter': 1000,
                    'random_state': 42
                },
                'mlp': {
                    'hidden_layer_sizes': (128, 64, 32),
                    'max_iter': 500,
                    'early_stopping': True,
                    'validation_fraction': 0.1,
                    'random_state': 42
                }
            },
            'evaluation': {
                'cross_validation': True,
                'cv_folds': 5,
                'test_size': 0.2
            }
        }
        
        if config_path and Path(config_path).exists():
            import yaml
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                default_config.update(user_config)
                
        return default_config
    
    def load_statistics_data(self, data_path: str) -> Tuple[Dict, Dict, Dict]:
        """加载统计量数据"""
        print(f"加载统计量数据: {data_path}")
        
        train_data = {}
        val_data = {}
        test_data = {}
        
        with h5py.File(data_path, 'r') as f:
            # 加载训练集
            if 'train' in f:
                train_data = {
                    'stokes_stats': f['train/stokes_stats'][:],
                    'fluorescence_stats': f['train/fluorescence_stats'][:],
                    'labels': f['train/labels'][:]
                }
                
            # 加载验证集
            if 'val' in f:
                val_data = {
                    'stokes_stats': f['val/stokes_stats'][:],
                    'fluorescence_stats': f['val/fluorescence_stats'][:],
                    'labels': f['val/labels'][:]
                }
                
            # 加载测试集
            if 'test' in f:
                test_data = {
                    'stokes_stats': f['test/stokes_stats'][:],
                    'fluorescence_stats': f['test/fluorescence_stats'][:],
                    'labels': f['test/labels'][:]
                }
        
        print(f"训练集: {len(train_data.get('labels', []))} 样本")
        print(f"验证集: {len(val_data.get('labels', []))} 样本") 
        print(f"测试集: {len(test_data.get('labels', []))} 样本")
        
        return train_data, val_data, test_data
    
    def prepare_features(self, data: Dict) -> np.ndarray:
        """准备特征向量"""
        if not data:
            return np.array([])
            
        # 展平统计量特征
        stokes_features = data['stokes_stats'].reshape(len(data['stokes_stats']), -1)
        fluorescence_features = data['fluorescence_stats'].reshape(len(data['fluorescence_stats']), -1)
        
        # 合并特征
        features = np.concatenate([stokes_features, fluorescence_features], axis=1)
        
        print(f"特征维度: {features.shape}")
        print(f"Stokes特征: {stokes_features.shape[1]} 维")
        print(f"荧光特征: {fluorescence_features.shape[1]} 维")
        
        return features
    
    def preprocess_features(self, X_train: np.ndarray, y_train: np.ndarray = None,
                          X_val: np.ndarray = None, X_test: np.ndarray = None) -> Tuple[np.ndarray, ...]:
        """特征预处理"""
        print("\n特征预处理...")
        
        # 0. 处理NaN值
        from sklearn.impute import SimpleImputer
        nan_count = np.isnan(X_train).sum()
        print(f"训练集NaN值数量: {nan_count}")
        
        if nan_count > 0:
            self.imputer = SimpleImputer(strategy='mean')
            X_train = self.imputer.fit_transform(X_train)
            print(f"已使用均值填充NaN值")
            
            if X_val is not None:
                X_val = self.imputer.transform(X_val)
            if X_test is not None:
                X_test = self.imputer.transform(X_test)
        
        # 1. 特征缩放
        scaler_type = self.config['preprocessing']['scaler']
        if scaler_type == 'standard':
            self.scalers['feature'] = StandardScaler()
        elif scaler_type == 'minmax':
            self.scalers['feature'] = MinMaxScaler()
        elif scaler_type == 'robust':
            self.scalers['feature'] = RobustScaler()
        
        X_train_scaled = self.scalers['feature'].fit_transform(X_train)
        print(f"使用 {scaler_type} 缩放器")
        
        results = [X_train_scaled]
        
        if X_val is not None:
            X_val_scaled = self.scalers['feature'].transform(X_val)
            results.append(X_val_scaled)
        
        if X_test is not None:
            X_test_scaled = self.scalers['feature'].transform(X_test)
            results.append(X_test_scaled)
        
        # 2. 特征选择
        if self.config['preprocessing']['feature_selection']:
            n_features = self.config['preprocessing']['n_features']
            self.feature_selector = SelectKBest(f_classif, k=min(n_features, X_train_scaled.shape[1]))
            
            results[0] = self.feature_selector.fit_transform(results[0], y_train)
            
            for i in range(1, len(results)):
                results[i] = self.feature_selector.transform(results[i])
                
            print(f"特征选择: 保留前 {min(n_features, X_train_scaled.shape[1])} 个特征")
        
        # 3. PCA降维
        if self.config['preprocessing']['pca']:
            n_components = self.config['preprocessing']['pca_components']
            self.pca = PCA(n_components=min(n_components, results[0].shape[1]))
            
            results[0] = self.pca.fit_transform(results[0])
            
            for i in range(1, len(results)):
                results[i] = self.pca.transform(results[i])
                
            print(f"PCA降维: {n_components} 主成分")
            print(f"解释方差比: {self.pca.explained_variance_ratio_.sum():.3f}")
        
        print(f"最终特征维度: {results[0].shape[1]}")
        
        return tuple(results)
    
    def create_models(self) -> Dict:
        """创建机器学习模型"""
        models = {}
        
        # 随机森林
        if 'random_forest' in self.config['models']:
            models['RandomForest'] = RandomForestClassifier(
                **self.config['models']['random_forest']
            )
        
        # 梯度提升
        if 'gradient_boosting' in self.config['models']:
            models['GradientBoosting'] = GradientBoostingClassifier(
                **self.config['models']['gradient_boosting']
            )
        
        # 支持向量机
        if 'svm' in self.config['models']:
            svm_config = self.config['models']['svm'].copy()
            # 确保probability参数只设置一次
            svm_config['probability'] = True
            models['SVM'] = SVC(**svm_config)
        
        # 逻辑回归
        if 'logistic_regression' in self.config['models']:
            models['LogisticRegression'] = LogisticRegression(
                **self.config['models']['logistic_regression']
            )
        
        # 多层感知机
        if 'mlp' in self.config['models']:
            models['MLP'] = MLPClassifier(
                **self.config['models']['mlp']
            )
        
        # 其他经典模型
        models['KNN'] = KNeighborsClassifier(n_neighbors=5)
        models['NaiveBayes'] = GaussianNB()
        models['ExtraTrees'] = ExtraTreesClassifier(
            n_estimators=200, random_state=42
        )
        
        print(f"创建了 {len(models)} 个模型: {list(models.keys())}")
        
        return models
    
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                    X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict:
        """训练模型"""
        print("\n开始训练模型...")
        
        models = self.create_models()
        results = {}
        
        for name, model in models.items():
            print(f"\n训练 {name}...")
            start_time = time.time()
            
            try:
                # 训练模型
                model.fit(X_train, y_train)
                
                # 训练集预测
                train_pred = model.predict(X_train)
                train_acc = accuracy_score(y_train, train_pred)
                
                # 验证集预测（如果有）
                val_acc = None
                if X_val is not None and y_val is not None:
                    val_pred = model.predict(X_val)
                    val_acc = accuracy_score(y_val, val_pred)
                
                # 交叉验证
                cv_scores = None
                if self.config['evaluation']['cross_validation']:
                    cv_folds = self.config['evaluation']['cv_folds']
                    cv_scores = cross_val_score(
                        model, X_train, y_train, 
                        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
                        scoring='accuracy'
                    )
                
                training_time = time.time() - start_time
                
                # 保存结果
                results[name] = {
                    'model': model,
                    'train_accuracy': train_acc,
                    'val_accuracy': val_acc,
                    'cv_scores': cv_scores,
                    'cv_mean': cv_scores.mean() if cv_scores is not None else None,
                    'cv_std': cv_scores.std() if cv_scores is not None else None,
                    'training_time': training_time
                }
                
                print(f"  训练时间: {training_time:.2f}s")
                print(f"  训练准确率: {train_acc:.4f}")
                if val_acc is not None:
                    print(f"  验证准确率: {val_acc:.4f}")
                if cv_scores is not None:
                    print(f"  交叉验证: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                
            except Exception as e:
                print(f"  训练失败: {e}")
                results[name] = {'error': str(e)}
        
        self.models = {k: v['model'] for k, v in results.items() if 'model' in v}
        
        return results
    
    def create_ensemble(self, X_train: np.ndarray, y_train: np.ndarray) -> VotingClassifier:
        """创建集成模型"""
        print("\n创建集成模型...")
        
        # 选择表现最好的几个模型
        base_models = []
        
        if 'RandomForest' in self.models:
            base_models.append(('rf', self.models['RandomForest']))
        if 'GradientBoosting' in self.models:
            base_models.append(('gb', self.models['GradientBoosting']))
        if 'ExtraTrees' in self.models:
            base_models.append(('et', self.models['ExtraTrees']))
        if 'SVM' in self.models:
            base_models.append(('svm', self.models['SVM']))
        
        if len(base_models) >= 2:
            ensemble = VotingClassifier(
                estimators=base_models,
                voting='soft'  # 使用概率投票
            )
            
            print(f"集成模型包含: {[name for name, _ in base_models]}")
            ensemble.fit(X_train, y_train)
            
            return ensemble
        
        return None
    
    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray, 
                      model_name: str) -> Dict:
        """评估单个模型"""
        print(f"\n评估 {model_name}...")
        
        # 预测
        y_pred = model.predict(X_test)
        y_prob = None
        
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)
        
        # 计算指标
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted'
        )
        
        # AUC (多分类)
        auc = None
        if y_prob is not None:
            try:
                auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')
            except:
                pass
        
        # 分类报告
        report = classification_report(
            y_test, y_pred, 
            target_names=[self.class_names[i] for i in sorted(self.class_names.keys())],
            output_dict=True
        )
        
        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'probabilities': y_prob
        }
        
        print(f"  准确率: {accuracy:.4f}")
        print(f"  精确率: {precision:.4f}")
        print(f"  召回率: {recall:.4f}")
        print(f"  F1分数: {f1:.4f}")
        if auc:
            print(f"  AUC: {auc:.4f}")
        
        return results
    
    def save_results(self, train_results: Dict, test_results: Dict, 
                    output_dir: str) -> None:
        """保存结果"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存训练结果
        train_summary = {}
        for name, result in train_results.items():
            if 'error' not in result:
                train_summary[name] = {
                    'train_accuracy': result['train_accuracy'],
                    'val_accuracy': result['val_accuracy'],
                    'cv_mean': result['cv_mean'],
                    'cv_std': result['cv_std'],
                    'training_time': result['training_time']
                }
        
        with open(output_path / f'training_results_{timestamp}.json', 'w') as f:
            json.dump(train_summary, f, indent=2, default=str)
        
        # 保存测试结果
        test_summary = {}
        for name, result in test_results.items():
            test_summary[name] = {
                'accuracy': result['accuracy'],
                'precision': result['precision'],
                'recall': result['recall'],
                'f1_score': result['f1_score'],
                'auc': result['auc']
            }
        
        with open(output_path / f'test_results_{timestamp}.json', 'w') as f:
            json.dump(test_summary, f, indent=2, default=str)
        
        # 保存模型
        for name, model in self.models.items():
            model_path = output_path / f'{name}_{timestamp}.joblib'
            joblib.dump(model, model_path)
        
        # 保存预处理器
        if self.scalers:
            joblib.dump(self.scalers, output_path / f'scalers_{timestamp}.joblib')
        if hasattr(self, 'imputer') and self.imputer:
            joblib.dump(self.imputer, output_path / f'imputer_{timestamp}.joblib')
        if self.feature_selector:
            joblib.dump(self.feature_selector, output_path / f'feature_selector_{timestamp}.joblib')
        if self.pca:
            joblib.dump(self.pca, output_path / f'pca_{timestamp}.joblib')
        
        print(f"\n结果已保存到: {output_path}")
    
    def plot_results(self, test_results: Dict, output_dir: str) -> None:
        """绘制结果图表"""
        output_path = Path(output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. 模型性能对比
        models = list(test_results.keys())
        accuracies = [test_results[m]['accuracy'] for m in models]
        f1_scores = [test_results[m]['f1_score'] for m in models]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 准确率对比
        bars1 = ax1.bar(models, accuracies, color='skyblue', alpha=0.8)
        ax1.set_title('Model Accuracy Comparison')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        plt.setp(ax1.get_xticklabels(), rotation=45)
        
        # 添加数值标签
        for bar, acc in zip(bars1, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # F1分数对比
        bars2 = ax2.bar(models, f1_scores, color='lightcoral', alpha=0.8)
        ax2.set_title('Model F1-Score Comparison')
        ax2.set_ylabel('F1-Score')
        ax2.set_ylim(0, 1)
        plt.setp(ax2.get_xticklabels(), rotation=45)
        
        # 添加数值标签
        for bar, f1 in zip(bars2, f1_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{f1:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_path / f'model_comparison_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 最佳模型的混淆矩阵
        best_model = max(test_results.keys(), key=lambda k: test_results[k]['accuracy'])
        cm = test_results[best_model]['confusion_matrix']
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=[self.class_names[i] for i in sorted(self.class_names.keys())],
                   yticklabels=[self.class_names[i] for i in sorted(self.class_names.keys())])
        plt.title(f'Confusion Matrix - {best_model}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig(output_path / f'confusion_matrix_{best_model}_{timestamp}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"图表已保存到: {output_path}")
    
    def run_experiment(self, data_path: str, output_dir: str) -> None:
        """运行完整实验"""
        print("=" * 60)
        print("基于统计量的藻类分类实验")
        print("=" * 60)
        
        # 1. 加载数据
        train_data, val_data, test_data = self.load_statistics_data(data_path)
        
        # 2. 准备特征
        X_train = self.prepare_features(train_data)
        y_train = train_data['labels']
        
        X_val = self.prepare_features(val_data) if val_data else None
        y_val = val_data['labels'] if val_data else None
        
        X_test = self.prepare_features(test_data)
        y_test = test_data['labels']
        
        # 3. 特征预处理
        if X_val is not None:
            X_train, X_val, X_test = self.preprocess_features(X_train, y_train, X_val, X_test)
        else:
            X_train, X_test = self.preprocess_features(X_train, y_train, X_test=X_test)
        
        # 4. 训练模型
        train_results = self.train_models(X_train, y_train, X_val, y_val)
        
        # 5. 创建集成模型
        ensemble = self.create_ensemble(X_train, y_train)
        if ensemble:
            self.models['Ensemble'] = ensemble
        
        # 6. 测试评估
        test_results = {}
        for name, model in self.models.items():
            test_results[name] = self.evaluate_model(model, X_test, y_test, name)
        
        # 7. 保存结果
        self.save_results(train_results, test_results, output_dir)
        
        # 8. 生成图表
        self.plot_results(test_results, output_dir)
        
        # 9. 打印总结
        print("\n" + "=" * 60)
        print("实验总结")
        print("=" * 60)
        
        print("\n最佳模型性能:")
        best_model = max(test_results.keys(), key=lambda k: test_results[k]['accuracy'])
        best_result = test_results[best_model]
        
        print(f"模型: {best_model}")
        print(f"准确率: {best_result['accuracy']:.4f}")
        print(f"F1分数: {best_result['f1_score']:.4f}")
        if best_result['auc']:
            print(f"AUC: {best_result['auc']:.4f}")
        
        print(f"\n结果已保存到: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='统计量机器学习分类器')
    parser.add_argument('--data', type=str, 
                       default='data/processed/multimodal_statistics.h5',
                       help='统计量HDF5文件路径')
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径')
    parser.add_argument('--output', type=str,
                       default='experiments/statistics_classification',
                       help='输出目录')
    
    args = parser.parse_args()
    
    # 创建分类器
    classifier = StatisticsClassifier(args.config)
    
    # 运行实验
    classifier.run_experiment(args.data, args.output)


if __name__ == "__main__":
    main()
