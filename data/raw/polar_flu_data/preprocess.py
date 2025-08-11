import os
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import pickle
import json
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import scipy.io
import warnings
import h5py
warnings.filterwarnings('ignore')

class PolarFluDataPreprocessor:
    """偏振荧光数据预处理器"""
    
    def __init__(self, raw_data_dir, output_dir):
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 数据类别映射
        self.class_mapping = {
            'CG': 0,      # 
            'IG': 1,      # 
            'PS3': 2,     # 3μm聚苯乙烯球
            'PS6': 3,     # 6μm聚苯乙烯球
            'PS10': 4,    # 10μm聚苯乙烯球
            'QDDB': 5,    # 
            'QZQG': 6,    # 
            'SG': 7,      # 
            'TP': 8,      #
            'TS': 9,      # 
            'YMXH': 10,   # 
            'YXXB': 11    # 
        }
        
        # 数据结构定义
        self.signal_length = 4000  # 保持原始信号长度
        self.rows_per_sample = 20  # 每个样本20行
        
        # 波长信息
        self.wavelengths = [445, 473, 520, 630]  # nm
        
        print(f"原始数据目录: {self.raw_data_dir}")
        print(f"输出目录: {self.output_dir}")
        print(f"信号长度: {self.signal_length}")
    
    def parse_mat_file_structure(self, mat_file_path):
        """解析.mat文件结构，支持v7.3格式"""
        print(f"\n分析文件: {mat_file_path}")
        
        try:
            # 首先尝试使用scipy.io读取
            try:
                mat_data = scipy.io.loadmat(mat_file_path)
                print("使用 scipy.io 成功读取文件")
                
                # 过滤掉元数据
                data_keys = [k for k in mat_data.keys() if not k.startswith('__')]
                
            except NotImplementedError as e:
                if "Please use HDF reader for matlab v7.3 files" in str(e):
                    print("检测到 MATLAB v7.3 格式，切换到 h5py 读取...")
                    
                    # 使用h5py读取v7.3格式文件
                    import h5py
                    
                    with h5py.File(mat_file_path, 'r') as f:
                        # 获取所有数据变量名
                        data_keys = list(f.keys())
                        
                        # 创建一个字典来存储数据
                        mat_data = {}
                        for key in data_keys:
                            if key.startswith('#'):  # 跳过HDF5元数据
                                continue
                            
                            dataset = f[key]
                            if isinstance(dataset, h5py.Dataset):
                                # 对于v7.3格式，数据可能需要转置
                                data = dataset[()]
                                if len(data.shape) == 2:
                                    # MATLAB以列优先存储，Python以行优先，需要转置
                                    data = data.T
                                mat_data[key] = data
                                print(f"从HDF5读取: {key}, 形状: {data.shape}")
                    
                    # 过滤有效的数据键
                    data_keys = [k for k in data_keys if k in mat_data and not k.startswith('#')]
                else:
                    raise e
            
            print(f"数据变量: {data_keys}")
            
            for key in data_keys:
                data = mat_data[key]
                print(f"  {key}: 形状 {data.shape}, 类型 {data.dtype}")
                
                # 分析数据是否符合预期格式
                if len(data.shape) == 2:
                    rows, cols = data.shape
                    expected_samples = rows // self.rows_per_sample
                    print(f"    预计样本数: {expected_samples}")
                    print(f"    信号长度: {cols}")
                    
                    if cols == self.signal_length:
                        print(f"    ✓ 信号长度匹配预期 ({self.signal_length})")
                    else:
                        print(f"    ⚠ 信号长度不匹配，预期 {self.signal_length}，实际 {cols}")
                    
                    if rows % self.rows_per_sample == 0:
                        print(f"    ✓ 行数符合每样本{self.rows_per_sample}行的格式")
                    else:
                        print(f"    ⚠ 行数不是{self.rows_per_sample}的倍数")
            
            return mat_data, data_keys
            
        except Exception as e:
            print(f"读取.mat文件时出错: {e}")
            return None, None
    
    def extract_signals_from_mat(self, mat_data, data_key):
        """从.mat数据中提取信号"""
        print(f"\n从 {data_key} 提取信号...")
        
        data = mat_data[data_key]
        rows, cols = data.shape
        
        # 计算样本数
        num_samples = rows // self.rows_per_sample
        print(f"检测到 {num_samples} 个样本")
        
        if num_samples == 0:
            print("错误: 没有检测到有效样本")
            return None, None, None
        
        # 初始化数组
        stokes_data = np.zeros((num_samples, 4, cols))  # [样本数, 4, 信号长度]
        excitation_data = np.zeros((num_samples, 4, cols))  # [样本数, 4波长, 信号长度]
        fluorescence_data = np.zeros((num_samples, 12, cols))  # [样本数, 12通道, 信号长度]
        
        for sample_idx in range(num_samples):
            start_row = sample_idx * self.rows_per_sample
            sample_data = data[start_row:start_row + self.rows_per_sample, :]
            
            # 提取各种信号
            # 第1-4行: I, Q, U, V (Stokes参数)
            stokes_data[sample_idx] = sample_data[0:4, :]
            
            # 第5-8行: 445, 473, 520, 630nm激发光信号
            excitation_data[sample_idx] = sample_data[4:8, :]
            
            # 第9-12行: F1 (445, 473, 520, 630nm)
            fluorescence_data[sample_idx, 0:4] = sample_data[8:12, :]
            
            # 第13-16行: F2 (445, 473, 520, 630nm)
            fluorescence_data[sample_idx, 4:8] = sample_data[12:16, :]
            
            # 第17-20行: F3 (445, 473, 520, 630nm)
            fluorescence_data[sample_idx, 8:12] = sample_data[16:20, :]
        
        print(f"提取完成:")
        print(f"  Stokes数据: {stokes_data.shape}")
        print(f"  激发光数据: {excitation_data.shape}")
        print(f"  荧光数据: {fluorescence_data.shape}")
        
        # 添加Stokes参数的详细统计信息
        print(f"\n  Stokes参数统计 (前10个样本):")
        stokes_names = ['I', 'Q', 'U', 'V']
        for i, name in enumerate(stokes_names):
            sample_means = [np.mean(stokes_data[j, i]) for j in range(min(10, stokes_data.shape[0]))]
            overall_mean = np.mean(stokes_data[:, i])
            overall_std = np.std(stokes_data[:, i])
            overall_range = [np.min(stokes_data[:, i]), np.max(stokes_data[:, i])]
            print(f"    {name}: 总体均值={overall_mean:.6f}, 标准差={overall_std:.6f}, 范围={overall_range}")
            print(f"       前10个样本均值: {[f'{m:.6f}' for m in sample_means]}")
        
        return stokes_data, excitation_data, fluorescence_data
    
    def process_class_data(self, class_name):
        """处理特定类别的数据"""
        print(f"\n{'='*50}")
        print(f"处理 {class_name} 类别数据")
        print(f"{'='*50}")
        
        # 直接在原始数据目录中查找以类别名开头的.mat文件
        mat_files = list(self.raw_data_dir.glob(f"{class_name}.mat"))
        if not mat_files:
            print(f"警告: 在 {self.raw_data_dir} 中没有找到 {class_name}.mat 文件")
            return None, None, None
        
        print(f"找到 {len(mat_files)} 个.mat文件")
        
        all_stokes = []
        all_fluorescence = []
        all_excitation = []
        
        for mat_file in mat_files:
            print(f"\n处理文件: {mat_file.name}")
            
            # 解析文件结构
            mat_data, data_keys = self.parse_mat_file_structure(mat_file)
            if mat_data is None:
                continue
            
            # 处理每个数据变量
            for data_key in data_keys:
                print(f"\n处理数据变量: {data_key}")
                
                stokes, excitation, fluorescence = self.extract_signals_from_mat(mat_data, data_key)
                
                if stokes is not None:
                    # 直接使用原始数据，不进行重采样
                    all_stokes.append(stokes)
                    all_fluorescence.append(fluorescence)
                    all_excitation.append(excitation)
        
        # 合并所有文件的数据
        if all_stokes:
            combined_stokes = np.concatenate(all_stokes, axis=0)
            combined_fluorescence = np.concatenate(all_fluorescence, axis=0)
            combined_excitation = np.concatenate(all_excitation, axis=0)
            
            print(f"\n{class_name} 类别数据合并完成:")
            print(f"  总样本数: {combined_stokes.shape[0]}")
            print(f"  Stokes数据: {combined_stokes.shape}")
            print(f"  荧光数据: {combined_fluorescence.shape}")
            print(f"  激发光数据: {combined_excitation.shape}")
            
            return combined_stokes, combined_fluorescence, combined_excitation
        else:
            print(f"{class_name} 没有有效数据")
            return None, None, None
    
    def create_dataset(self):
        """创建完整数据集"""
        print(f"\n{'='*60}")
        print("开始创建完整数据集")
        print(f"{'='*60}")
        
        all_stokes = []
        all_fluorescence = []
        all_excitation = []
        all_labels = []
        sample_counts = {}
        
        for class_name, label in self.class_mapping.items():
            stokes, fluorescence, excitation = self.process_class_data(class_name)
            
            if stokes is not None:
                num_samples = stokes.shape[0]
                sample_counts[class_name] = num_samples
                
                # 创建标签
                labels = np.full(num_samples, label)
                
                all_stokes.append(stokes)
                all_fluorescence.append(fluorescence)
                all_excitation.append(excitation)
                all_labels.append(labels)
            else:
                sample_counts[class_name] = 0
        
        if not all_stokes:
            print("错误: 没有找到任何有效数据")
            return None
        
        # 合并所有类别数据
        final_stokes = np.concatenate(all_stokes, axis=0)
        final_fluorescence = np.concatenate(all_fluorescence, axis=0)
        final_excitation = np.concatenate(all_excitation, axis=0)
        final_labels = np.concatenate(all_labels, axis=0)
        
        print(f"\n{'='*60}")
        print("数据集创建完成")
        print(f"{'='*60}")
        print(f"总样本数: {len(final_labels)}")
        print(f"Stokes数据形状: {final_stokes.shape}")
        print(f"荧光数据形状: {final_fluorescence.shape}")
        print(f"激发光数据形状: {final_excitation.shape}")
        print(f"标签形状: {final_labels.shape}")
        
        print(f"\n各类别样本分布:")
        for class_name, count in sample_counts.items():
            print(f"  {class_name}: {count} 样本")
        
        return {
            'stokes_data': final_stokes,
            'fluorescence_data': final_fluorescence,
            'excitation_data': final_excitation,
            'labels': final_labels,
            'sample_counts': sample_counts
        }
    
    def save_processed_data(self, dataset):
        """保存处理后的数据"""
        print(f"\n保存处理后的数据到: {self.output_dir}")
        
        # 创建数据目录
        data_dir = self.output_dir / 'processed_data'
        data_dir.mkdir(exist_ok=True)

         # 创建类别数据目录
        class_data_dir = data_dir / 'class_data'
        class_data_dir.mkdir(exist_ok=True)
        
        # 保存主要数据
        np.save(data_dir / 'stokes_data.npy', dataset['stokes_data'])
        np.save(data_dir / 'fluorescence_data.npy', dataset['fluorescence_data'])
        np.save(data_dir / 'excitation_data.npy', dataset['excitation_data'])
        np.save(data_dir / 'labels.npy', dataset['labels'])
        
        print("数据文件保存完成:")
        print(f"  - stokes_data.npy: {dataset['stokes_data'].shape}")
        print(f"  - fluorescence_data.npy: {dataset['fluorescence_data'].shape}")
        print(f"  - excitation_data.npy: {dataset['excitation_data'].shape}")
        print(f"  - labels.npy: {dataset['labels'].shape}")
        
         # 分别保存每个类别的数据
        print(f"\n分别保存各类别数据到: {class_data_dir}")
    
        stokes_data = dataset['stokes_data']
        fluorescence_data = dataset['fluorescence_data']
        excitation_data = dataset['excitation_data']
        labels = dataset['labels']
    
        class_file_info = {}
    
        for class_name, label in self.class_mapping.items():
            if dataset['sample_counts'][class_name] == 0:
                print(f"  跳过 {class_name} (无数据)")
                continue
            
        # 获取该类别的样本索引
            class_mask = labels == label
            class_indices = np.where(class_mask)[0]
        
            if len(class_indices) == 0:
                continue
            
            print(f"  保存 {class_name} 类别数据 ({len(class_indices)} 个样本)...")
        
            # 提取该类别的数据
            class_stokes = stokes_data[class_mask]
            class_fluorescence = fluorescence_data[class_mask]
            class_excitation = excitation_data[class_mask]
            class_labels = labels[class_mask]
        
            # 创建类别子目录
            class_subdir = class_data_dir / class_name
            class_subdir.mkdir(exist_ok=True)
        
            # 保存该类别的数据
            np.save(class_subdir / f'{class_name}_stokes_data.npy', class_stokes)
            np.save(class_subdir / f'{class_name}_fluorescence_data.npy', class_fluorescence)
            np.save(class_subdir / f'{class_name}_excitation_data.npy', class_excitation)
            np.save(class_subdir / f'{class_name}_labels.npy', class_labels)
        
            # 保存该类别的元数据
            class_metadata = {
                'class_name': class_name,
                'class_label': label,
                'sample_count': len(class_indices),
                'data_shapes': {
                    'stokes': list(class_stokes.shape),
                    'fluorescence': list(class_fluorescence.shape),
                    'excitation': list(class_excitation.shape),
                    'labels': list(class_labels.shape)
                },
                'sample_indices': class_indices.tolist(),
                'data_statistics': {
                    'stokes': {
                        'mean': float(np.mean(class_stokes)),
                        'std': float(np.std(class_stokes)),
                        'min': float(np.min(class_stokes)),
                        'max': float(np.max(class_stokes))
                    },
                    'fluorescence': {
                        'mean': float(np.mean(class_fluorescence)),
                        'std': float(np.std(class_fluorescence)),
                        'min': float(np.min(class_fluorescence)),
                        'max': float(np.max(class_fluorescence))
                    },
                    'excitation': {
                        'mean': float(np.mean(class_excitation)),
                        'std': float(np.std(class_excitation)),
                        'min': float(np.min(class_excitation)),
                        'max': float(np.max(class_excitation))
                    }
             }
         }
        
        with open(class_subdir / f'{class_name}_metadata.json', 'w', encoding='utf-8') as f:
            json.dump(class_metadata, f, indent=2, ensure_ascii=False)
        
        # 记录文件信息
        class_file_info[class_name] = {
            'directory': str(class_subdir),
            'sample_count': len(class_indices),
            'files': [
                f'{class_name}_stokes_data.npy',
                f'{class_name}_fluorescence_data.npy',
                f'{class_name}_excitation_data.npy',
                f'{class_name}_labels.npy',
                f'{class_name}_metadata.json'
            ]
        }
        
        print(f"    - 保存到: {class_subdir}")
        print(f"    - Stokes数据: {class_stokes.shape}")
        print(f"    - 荧光数据: {class_fluorescence.shape}")
        print(f"    - 激发光数据: {class_excitation.shape}")
    
        # 保存总体元数据
        metadata = {
            'class_mapping': self.class_mapping,
            'sample_counts': dataset['sample_counts'],
            'data_shapes': {
                'stokes': list(dataset['stokes_data'].shape),
                'fluorescence': list(dataset['fluorescence_data'].shape),
                'excitation': list(dataset['excitation_data'].shape),
                'labels': list(dataset['labels'].shape)
            },
            'signal_info': {
                'signal_length': self.signal_length,
                'rows_per_sample': self.rows_per_sample,
                'wavelengths': self.wavelengths
            },
            'data_structure': {
                'stokes_channels': ['I', 'Q', 'U', 'V'],
                'wavelengths': self.wavelengths,
                'fluorescence_channels': [
                    'F1_445nm', 'F1_473nm', 'F1_520nm', 'F1_630nm',
                    'F2_445nm', 'F2_473nm', 'F2_520nm', 'F2_630nm',
                    'F3_445nm', 'F3_473nm', 'F3_520nm', 'F3_630nm'
                ]
            },
            'class_file_info': class_file_info
        }
    
        with open(data_dir / 'metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
        print(f"\n元数据保存到: {data_dir / 'metadata.json'}")
        print(f"类别数据保存到: {class_data_dir}")
        
        return data_dir
    
    def create_visualization(self, dataset):
        """创建数据可视化"""
        print(f"\n创建数据可视化...")
        
        vis_dir = self.output_dir / 'visualization'
        vis_dir.mkdir(exist_ok=True)
        
        stokes_data = dataset['stokes_data']
        fluorescence_data = dataset['fluorescence_data']
        excitation_data = dataset['excitation_data']
        labels = dataset['labels']
        
        # 为每个类别创建示例图
        for class_name, label in self.class_mapping.items():
            if dataset['sample_counts'][class_name] == 0:
                continue
            
            # 找到该类别的样本
            class_mask = labels == label
            class_indices = np.where(class_mask)[0]
            
            if len(class_indices) == 0:
                continue
            
            # 随机选择一个样本
            sample_idx = np.random.choice(class_indices)
            
            # 创建图像
            fig, axes = plt.subplots(3, 2, figsize=(15, 12))
            fig.suptitle(f'{class_name} Sample Example (Sample #{sample_idx})', fontsize=16)
            
            # Stokes parameters
            stokes_sample = stokes_data[sample_idx]
            stokes_names = ['I', 'Q', 'U', 'V']
            colors = ['red', 'blue', 'green', 'orange']
            
            # 添加数据诊断信息
            print(f"\n{class_name} 样本 {sample_idx} 的Stokes参数统计:")
            for idx, name in enumerate(stokes_names):
                mean_val = np.mean(stokes_sample[idx])
                std_val = np.std(stokes_sample[idx])
                min_val = np.min(stokes_sample[idx])
                max_val = np.max(stokes_sample[idx])
                print(f"  {name}: 均值={mean_val:.6f}, 标准差={std_val:.6f}, 范围=[{min_val:.6f}, {max_val:.6f}]")
            
            for i, (name, color) in enumerate(zip(stokes_names, colors)):
                axes[0, 0].plot(stokes_sample[i], label=name, color=color, alpha=0.8)
            axes[0, 0].set_title('Stokes Parameters (I, Q, U, V)')
            axes[0, 0].set_xlabel('Sample Point')
            axes[0, 0].set_ylabel('Intensity')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
           
            
            # 激发光信号
            excitation_sample = excitation_data[sample_idx]
            wavelength_colors = ['purple', 'blue', 'green', 'red']
            
            for i, (wl, color) in enumerate(zip(self.wavelengths, wavelength_colors)):
                axes[0, 1].plot(excitation_sample[i], label=f'{wl}nm', color=color, alpha=0.8)
            axes[0, 1].set_title('Excitation Signals')
            axes[0, 1].set_xlabel('Sample Point')
            axes[0, 1].set_ylabel('Intensity')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # F1 Fluorescence
            fluorescence_sample = fluorescence_data[sample_idx]
            for i, (wl, color) in enumerate(zip(self.wavelengths, wavelength_colors)):
                axes[1, 0].plot(fluorescence_sample[i], label=f'F1_{wl}nm', color=color, alpha=0.8)
            axes[1, 0].set_title('F1 Fluorescence')
            axes[1, 0].set_xlabel('Sample Point')
            axes[1, 0].set_ylabel('Intensity')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # F2 Fluorescence
            for i, (wl, color) in enumerate(zip(self.wavelengths, wavelength_colors)):
                axes[1, 1].plot(fluorescence_sample[i+4], label=f'F2_{wl}nm', color=color, alpha=0.8)
            axes[1, 1].set_title('F2 Fluorescence')
            axes[1, 1].set_xlabel('Sample Point')
            axes[1, 1].set_ylabel('Intensity')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            # F3 Fluorescence
            for i, (wl, color) in enumerate(zip(self.wavelengths, wavelength_colors)):
                axes[2, 0].plot(fluorescence_sample[i+8], label=f'F3_{wl}nm', color=color, alpha=0.8)
            axes[2, 0].set_title('F3 Fluorescence')
            axes[2, 0].set_xlabel('Sample Point')
            axes[2, 0].set_ylabel('Intensity')
            axes[2, 0].legend()
            axes[2, 0].grid(True, alpha=0.3)
            
            # Mean Fluorescence Intensity per Wavelength
            f1_means = [np.mean(fluorescence_sample[i]) for i in range(4)]
            f2_means = [np.mean(fluorescence_sample[i+4]) for i in range(4)]
            f3_means = [np.mean(fluorescence_sample[i+8]) for i in range(4)]
            
            x = np.arange(len(self.wavelengths))
            width = 0.25
            
            axes[2, 1].bar(x - width, f1_means, width, label='F1', alpha=0.8)
            axes[2, 1].bar(x, f2_means, width, label='F2', alpha=0.8)
            axes[2, 1].bar(x + width, f3_means, width, label='F3', alpha=0.8)
            
            axes[2, 1].set_title('Mean Fluorescence Intensity by Wavelength')
            axes[2, 1].set_xlabel('Wavelength (nm)')
            axes[2, 1].set_ylabel('Mean Intensity')
            axes[2, 1].set_xticks(x)
            axes[2, 1].set_xticklabels([f'{wl}nm' for wl in self.wavelengths])
            axes[2, 1].legend()
            axes[2, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(vis_dir / f'{class_name}_sample_analysis.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        # 创建数据分布统计图
        self._create_distribution_plots(dataset, vis_dir)
        
        print(f"可视化文件保存到: {vis_dir}")
    
    def _create_distribution_plots(self, dataset, vis_dir):
        """创建数据分布统计图"""
        
        # 1. 类别分布饼图
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 样本数分布
        class_names = list(dataset['sample_counts'].keys())
        sample_counts = list(dataset['sample_counts'].values())
        
        # 过滤掉0样本的类别
        non_zero_classes = [(name, count) for name, count in zip(class_names, sample_counts) if count > 0]
        if non_zero_classes:
            names, counts = zip(*non_zero_classes)
            axes[0].pie(counts, labels=names, autopct='%1.1f%%', startangle=90)
            axes[0].set_title('Class Distribution')
        
        # Signal intensity distribution
        all_stokes = dataset['stokes_data'].flatten()
        all_fluorescence = dataset['fluorescence_data'].flatten()
        
        axes[1].hist(all_stokes, bins=50, alpha=0.5, label='Stokes', density=True)
        axes[1].hist(all_fluorescence, bins=50, alpha=0.5, label='Fluorescence', density=True)
        axes[1].set_title('Signal Intensity Distribution')
        axes[1].set_xlabel('Signal Intensity')
        axes[1].set_ylabel('Density')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(vis_dir / 'data_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. Class signal feature comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Class Signal Feature Comparison', fontsize=16)
        
        labels = dataset['labels']
        
        # Stokes parameter mean comparison
        for class_name, label in self.class_mapping.items():
            if dataset['sample_counts'][class_name] == 0:
                continue
            
            class_mask = labels == label
            if not np.any(class_mask):
                continue
            
            class_stokes = dataset['stokes_data'][class_mask]
            stokes_means = np.mean(class_stokes, axis=(0, 2))  # Mean over samples and time
            
            axes[0, 0].bar(np.arange(4) + label*0.1, stokes_means, 
                  width=0.1, label=class_name, alpha=0.8)
        
        axes[0, 0].set_title('Mean Stokes Parameters by Class')
        axes[0, 0].set_xlabel('Stokes Parameter')
        axes[0, 0].set_ylabel('Mean Intensity')
        axes[0, 0].set_xticks(range(4))
        axes[0, 0].set_xticklabels(['I', 'Q', 'U', 'V'])
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Mean fluorescence comparison (F1)
        for class_name, label in self.class_mapping.items():
            if dataset['sample_counts'][class_name] == 0:
                continue

            class_mask = labels == label
            if not np.any(class_mask):
                continue

            class_fluorescence = dataset['fluorescence_data'][class_mask]
            f1_means = np.mean(class_fluorescence[:, 0:4], axis=(0, 2))  # F1, 4 wavelengths

            axes[0, 1].plot(self.wavelengths, f1_means,
                    marker='o', label=class_name, alpha=0.8)

        axes[0, 1].set_title('F1 Fluorescence Response by Class')
        axes[0, 1].set_xlabel('Wavelength (nm)')
        axes[0, 1].set_ylabel('Mean Fluorescence Intensity')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Similarly plot F2 and F3
        for class_name, label in self.class_mapping.items():
            if dataset['sample_counts'][class_name] == 0:
                continue

            class_mask = labels == label
            if not np.any(class_mask):
                continue

            class_fluorescence = dataset['fluorescence_data'][class_mask]
            f2_means = np.mean(class_fluorescence[:, 4:8], axis=(0, 2))
            f3_means = np.mean(class_fluorescence[:, 8:12], axis=(0, 2))

            axes[1, 0].plot(self.wavelengths, f2_means,
                    marker='s', label=class_name, alpha=0.8)
            axes[1, 1].plot(self.wavelengths, f3_means,
                    marker='^', label=class_name, alpha=0.8)

        axes[1, 0].set_title('F2 Fluorescence Response by Class')
        axes[1, 0].set_xlabel('Wavelength (nm)')
        axes[1, 0].set_ylabel('Mean Fluorescence Intensity')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].set_title('F3 Fluorescence Response by Class')
        axes[1, 1].set_xlabel('Wavelength (nm)')
        axes[1, 1].set_ylabel('Mean Fluorescence Intensity')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(vis_dir / 'class_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def generate_report(self, dataset, data_dir):
        """生成处理报告"""
        report_path = self.output_dir / 'preprocessing_report.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("偏振荧光数据预处理报告\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"处理时间: {pd.Timestamp.now()}\n")
            f.write(f"原始数据目录: {self.raw_data_dir}\n")
            f.write(f"输出目录: {self.output_dir}\n\n")
            
            f.write("数据结构信息:\n")
            f.write(f"  - 信号长度: {self.signal_length}\n")
            f.write(f"  - 每样本行数: {self.rows_per_sample}\n")
            f.write(f"  - 激发波长: {self.wavelengths} nm\n\n")
            
            f.write("处理后数据统计:\n")
            f.write(f"  - 总样本数: {len(dataset['labels'])}\n")
            f.write(f"  - Stokes数据形状: {dataset['stokes_data'].shape}\n")
            f.write(f"  - 荧光数据形状: {dataset['fluorescence_data'].shape}\n")
            f.write(f"  - 激发光数据形状: {dataset['excitation_data'].shape}\n")
            f.write(f"  - 标签数据形状: {dataset['labels'].shape}\n\n")
            
            f.write("各类别样本分布:\n")
            for class_name, count in dataset['sample_counts'].items():
                percentage = (count / len(dataset['labels']) * 100) if count > 0 else 0
                f.write(f"  - {class_name}: {count} 样本 ({percentage:.1f}%)\n")
            f.write("\n")
            
            f.write("数据质量统计:\n")
            
            # Stokes数据统计
            stokes_data = dataset['stokes_data']
            f.write(f"  Stokes数据:\n")
            f.write(f"    - 均值: {np.mean(stokes_data):.6f}\n")
            f.write(f"    - 标准差: {np.std(stokes_data):.6f}\n")
            f.write(f"    - 最小值: {np.min(stokes_data):.6f}\n")
            f.write(f"    - 最大值: {np.max(stokes_data):.6f}\n\n")
            
            # 荧光数据统计
            fluorescence_data = dataset['fluorescence_data']
            f.write(f"  荧光数据:\n")
            f.write(f"    - 均值: {np.mean(fluorescence_data):.6f}\n")
            f.write(f"    - 标准差: {np.std(fluorescence_data):.6f}\n")
            f.write(f"    - 最小值: {np.min(fluorescence_data):.6f}\n")
            f.write(f"    - 最大值: {np.max(fluorescence_data):.6f}\n\n")
            
            f.write("输出文件:\n")
            f.write(f"  - 处理后数据: {data_dir}\n")
            f.write(f"  - 可视化图像: {self.output_dir / 'visualization'}\n")
            f.write(f"  - 元数据: {data_dir / 'metadata.json'}\n")
        
        print(f"处理报告保存到: {report_path}")
    
    def run_complete_preprocessing(self):
        """运行完整预处理流程"""
        print("开始偏振荧光数据预处理...\n")
        
        try:
            # 1. 创建数据集
            dataset = self.create_dataset()
            if dataset is None:
                print("错误: 数据集创建失败")
                return None
            
            # 2. 保存处理后的数据
            data_dir = self.save_processed_data(dataset)
            
            # 3. 创建可视化
            self.create_visualization(dataset)
            
            # 4. 生成报告
            self.generate_report(dataset, data_dir)
            
            print(f"\n{'='*60}")
            print("预处理完成!")
            print(f"{'='*60}")
            print(f"处理后数据保存在: {self.output_dir}")
            print(f"总样本数: {len(dataset['labels'])}")
            print("数据已准备就绪，可用于模型训练。")
            
            return dataset
            
        except Exception as e:
            print(f"预处理过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """主函数"""
    # 设置路径
    raw_data_dir = "/data3/zs/Multimodel_fusion/raw_data/polar_flu_data"
    output_dir = "/data3/zs/Multimodel_fusion/processed_data"
    
    # 创建预处理器
    preprocessor = PolarFluDataPreprocessor(raw_data_dir, output_dir)
    
    # 运行完整预处理
    dataset = preprocessor.run_complete_preprocessing()
    
    if dataset:
        print("\n预处理成功完成!")
        print("您现在可以使用处理后的数据进行模型训练。")
    else:
        print("\n预处理失败，请检查数据格式和路径。")


if __name__ == "__main__":
    main()