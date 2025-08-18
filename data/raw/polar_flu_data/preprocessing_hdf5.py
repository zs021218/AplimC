#!/usr/bin/env python3
"""
基于HDF5的多模态极化荧光数据预处理器
直接写入HDF5，避免内存合并问题
"""

from pathlib import Path
import numpy as np
import h5py
import json
from PIL import Image
from typing import Dict, List, Tuple, Optional, Any
import yaml
from sklearn.model_selection import train_test_split
import gc
import psutil
import os
import re

class MultimodalHDF5Preprocessor:
    """基于HDF5的多模态数据预处理器"""
    
    def __init__(self, config_path: str = "params.yaml"):
        self._load_config(config_path)
        
        # 类别映射
        self.class_map = {
            'CG': 0, 'IG': 1, 'PS3': 2, 'PS6': 3, 'PS10': 4, 'QDDB': 5,
            'QZQG': 6, 'SG': 7, 'TP': 8, 'TS': 9, 'YMXH': 10, 'YXXB': 11
        }
        
        # HDF5文件路径
        self.hdf5_file = self.output_path / 'multimodal_data.h5'
        
        print(f"初始化完成:")
        print(f"  原始数据路径: {self.raw_data_path}")
        print(f"  图像数据路径: {self.image_data_path}")
        print(f"  输出路径: {self.output_path}")
        print(f"  HDF5文件: {self.hdf5_file}")
        print(f"  批处理大小: {self.batch_size}")

    def _load_config(self, config_path: str) -> None:
        """加载配置文件"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            # 路径配置
            self.raw_data_path = Path(config['paths']['raw_data'])
            self.image_data_path = Path(config['paths']['image_data'])
            self.output_path = Path(config['paths']['output'])
            self.output_path.mkdir(parents=True, exist_ok=True)
            
            # 预处理参数
            preprocess = config.get('preprocess', {})
            self.signal_length = preprocess.get('signal_length', 4000)
            self.rows_per_sample = preprocess.get('rows_per_sample', 20)
            self.image_size = tuple(preprocess.get('image_size', [224, 224]))
            self.num_views = preprocess.get('num_views', 3)
            self.batch_size = preprocess.get('batch_size', 50)
            self.memory_threshold = preprocess.get('memory_threshold', 0.85)
            
            # 分割参数
            self.split_params = config.get('split', {
                'train_ratio': 0.7, 'val_ratio': 0.15, 'test_ratio': 0.15, 'random_state': 42
            })
            
        except Exception as e:
            print(f"配置加载失败: {e}")
            # 使用默认配置
            self._set_default_config()

    def _set_default_config(self) -> None:
        """设置默认配置"""
        self.raw_data_path = Path('/data3/zs/AplimC/data/raw/polar_flu_data')
        self.image_data_path = Path('/data3/zs/AplimC/data/raw/images')
        self.output_path = Path('/data3/zs/AplimC/data/processed')
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self.signal_length = 4000
        self.rows_per_sample = 20
        self.image_size = (224, 224)
        self.num_views = 3
        self.batch_size = 50
        self.memory_threshold = 0.85
        
        self.split_params = {'train_ratio': 0.7, 'val_ratio': 0.15, 'test_ratio': 0.15, 'random_state': 42}

    def _check_memory(self) -> None:
        """检查内存使用并执行垃圾回收"""
        try:
            process = psutil.Process(os.getpid())
            memory_percent = process.memory_percent()
            if memory_percent > self.memory_threshold * 100:
                print(f"内存使用率: {memory_percent:.1f}%, 执行垃圾回收")
                gc.collect()
        except:
            gc.collect()

    def _extract_signals(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """提取信号数据"""
        rows, cols = data.shape
        num_samples = rows // self.rows_per_sample
        
        stokes = np.zeros((num_samples, 4, cols), dtype=np.float32)
        fluorescence = np.zeros((num_samples, 16, cols), dtype=np.float32)
        
        for i in range(num_samples):
            start_row = i * self.rows_per_sample
            sample_data = data[start_row:start_row + 20, :]
            stokes[i] = sample_data[:4, :].astype(np.float32)
            fluorescence[i] = sample_data[4:20, :].astype(np.float32)
            
            if i % 100 == 0:
                self._check_memory()
                
        return stokes, fluorescence

    def _load_image(self, image_path: Path) -> Optional[np.ndarray]:
        """加载并预处理单张图像"""
        try:
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img = img.resize(self.image_size, Image.Resampling.LANCZOS)
                return np.array(img, dtype=np.float32) / 255.0
        except Exception as e:
            print(f"图像加载失败 {image_path}: {e}")
            return None

    def _process_class_images(self, class_name: str) -> Optional[np.ndarray]:
        """处理类别图像"""
        # 构建图像目录路径
        class_dir = self.image_data_path / f"{class_name}-three views"
        if not class_dir.exists():
            print(f"图像目录不存在: {class_dir}")
            return None
            
        # 获取三个视图目录
        view_dirs = []
        for i in range(1, self.num_views + 1):
            view_dir = class_dir / f"view{i}"
            if view_dir.exists():
                view_dirs.append(view_dir)
            else:
                print(f"视图目录不存在: {view_dir}")
                
        if len(view_dirs) != self.num_views:
            print(f"视图目录数量不匹配: 期望{self.num_views}, 实际{len(view_dirs)}")
            return None
            
        # 获取每个视图的图像文件
        view_images = {}
        for idx, view_dir in enumerate(view_dirs):
            images = []
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                images.extend(view_dir.glob(f"*{ext}"))
                images.extend(view_dir.glob(f"*{ext.upper()}"))
            
            # 按文件名中的数字排序
            images = sorted(images, key=lambda x: int(re.findall(r'\d+', x.name)[0]) if re.findall(r'\d+', x.name) else 0)
            view_images[idx] = images
            
        # 取最小样本数
        min_samples = min(len(images) for images in view_images.values())
        if min_samples == 0:
            return None
            
        # 预分配数组
        all_images = np.zeros((min_samples, self.num_views, *self.image_size, 3), dtype=np.float32)
        
        # 批量处理图像
        valid_samples = 0
        for sample_idx in range(min_samples):
            views_loaded = 0
            for view_idx in range(self.num_views):
                if sample_idx < len(view_images[view_idx]):
                    image_path = view_images[view_idx][sample_idx]
                    processed_image = self._load_image(image_path)
                    if processed_image is not None:
                        all_images[sample_idx, view_idx] = processed_image
                        views_loaded += 1
                        
            if views_loaded == self.num_views:
                valid_samples += 1
            else:
                print(f"样本 {sample_idx} 视图不完整")
                
            if sample_idx % 50 == 0:
                self._check_memory()
                
        return all_images[:valid_samples] if valid_samples > 0 else None

    def _process_class(self, class_name: str) -> Optional[dict]:
        """处理单个类别的数据"""
        print(f"\n处理类别: {class_name}")
        
        # 1. 处理信号数据
        mat_file = self.raw_data_path / f"{class_name}.mat"
        if not mat_file.exists():
            print(f"信号文件不存在: {mat_file}")
            return None
            
        try:
            with h5py.File(mat_file, 'r') as f:
                data_keys = [k for k in f.keys() if not k.startswith('#')]
                if not data_keys:
                    print(f"未找到有效数据键: {mat_file}")
                    return None
                    
                data = f[data_keys[0]][()]
                if len(data.shape) == 2:
                    data = data.T
                    
                data = data.astype(np.float32)
                
        except Exception as e:
            print(f"信号文件加载失败 {mat_file}: {e}")
            return None
            
        stokes, fluorescence = self._extract_signals(data)
        del data
        self._check_memory()
        
        # 2. 处理图像数据
        images = self._process_class_images(class_name)
        if images is None:
            print(f"类别 {class_name} 图像处理失败")
            return None
            
        # 3. 数据对齐
        signal_samples = len(stokes)
        image_samples = len(images)
        aligned_samples = min(signal_samples, image_samples)
        
        if aligned_samples == 0:
            print(f"类别 {class_name} 没有有效样本")
            return None
            
        # 创建标签
        labels = np.full(aligned_samples, self.class_map[class_name], dtype=np.int32)
        
        result = {
            'stokes': stokes[:aligned_samples],
            'fluorescence': fluorescence[:aligned_samples],
            'images': images[:aligned_samples],
            'labels': labels,
            'class_name': class_name
        }
        
        print(f"类别 {class_name}: {aligned_samples} 样本")
        self._check_memory()
        
        return result

    def _create_hdf5_structure(self, total_samples: int) -> h5py.File:
        """创建HDF5文件结构"""
        if self.hdf5_file.exists():
            self.hdf5_file.unlink()
            
        f = h5py.File(self.hdf5_file, 'w')
        
        # 创建数据集 - 使用chunking优化，无压缩以提高写入速度
        chunk_size = min(100, total_samples)
        
        # 完整数据集组
        full_group = f.create_group('full')
        full_group.create_dataset('stokes', (total_samples, 4, self.signal_length), 
                                 dtype=np.float32, chunks=(chunk_size, 4, self.signal_length))
        full_group.create_dataset('fluorescence', (total_samples, 16, self.signal_length), 
                                 dtype=np.float32, chunks=(chunk_size, 16, self.signal_length))
        full_group.create_dataset('images', (total_samples, self.num_views, *self.image_size, 3), 
                                 dtype=np.float32, chunks=(chunk_size, self.num_views, *self.image_size, 3))
        full_group.create_dataset('labels', (total_samples,), 
                                 dtype=np.int32, chunks=(chunk_size,))
        
        # 分割组 - 先创建空组，稍后添加数据集
        for split_name in ['train', 'val', 'test']:
            f.create_group(split_name)
            
        return f

    def _write_split_to_hdf5(self, f: h5py.File, split_name: str, indices: np.ndarray) -> None:
        """将分割数据写入HDF5"""
        print(f"写入 {split_name} 集 ({len(indices)} 样本)...")
        
        # 对索引进行排序，HDF5要求索引递增
        sorted_indices = np.sort(indices)
        
        split_group = f[split_name]
        full_group = f['full']
        
        # 创建分割数据集 - 无压缩以提高写入速度
        chunk_size = min(100, len(indices))
        
        split_group.create_dataset('stokes', (len(indices), 4, self.signal_length), 
                                  dtype=np.float32, chunks=(chunk_size, 4, self.signal_length))
        split_group.create_dataset('fluorescence', (len(indices), 16, self.signal_length), 
                                  dtype=np.float32, chunks=(chunk_size, 16, self.signal_length))
        split_group.create_dataset('images', (len(indices), self.num_views, *self.image_size, 3), 
                                  dtype=np.float32, chunks=(chunk_size, self.num_views, *self.image_size, 3))
        split_group.create_dataset('labels', (len(indices),), 
                                  dtype=np.int32, chunks=(chunk_size,))
        
        # 批量复制数据
        batch_size = min(self.batch_size, len(sorted_indices))
        for i in range(0, len(sorted_indices), batch_size):
            end_i = min(i + batch_size, len(sorted_indices))
            batch_indices = sorted_indices[i:end_i]
            
            # 复制数据
            split_group['stokes'][i:end_i] = full_group['stokes'][batch_indices]
            split_group['fluorescence'][i:end_i] = full_group['fluorescence'][batch_indices]
            split_group['images'][i:end_i] = full_group['images'][batch_indices]
            split_group['labels'][i:end_i] = full_group['labels'][batch_indices]
            
            self._check_memory()

    def _save_metadata(self, total_samples: int, sample_counts: dict) -> None:
        """保存元数据"""
        metadata = {
            'class_map': self.class_map,
            'total_samples': total_samples,
            'sample_counts': sample_counts,
            'signal_length': self.signal_length,
            'image_size': list(self.image_size),
            'num_views': self.num_views,
            'split_params': self.split_params
        }
        
        with open(self.output_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

    def _generate_metrics(self, f: h5py.File) -> None:
        """生成分割统计信息"""
        metrics = {}
        
        for split_name in ['train', 'val', 'test']:
            if split_name in f:
                labels = f[split_name]['labels'][:]
                total = len(labels)
                
                class_stats = {}
                for class_name, class_label in self.class_map.items():
                    count = int(np.sum(labels == class_label))
                    percentage = float(count / total * 100) if total > 0 else 0.0
                    class_stats[class_name] = {'count': count, 'percentage': percentage}
                
                metrics[split_name] = {'total_samples': total, 'class_stats': class_stats}
        
        with open(self.output_path / 'split_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

    def process(self) -> bool:
        """主处理流程"""
        print("开始多模态数据预处理 (HDF5版本)...")
        
        # 1. 收集所有类别数据统计
        all_data = []
        sample_counts = {}
        total_samples = 0
        
        for class_name in self.class_map.keys():
            class_data = self._process_class(class_name)
            if class_data is not None:
                all_data.append(class_data)
                sample_counts[class_name] = len(class_data['labels'])
                total_samples += len(class_data['labels'])
            else:
                sample_counts[class_name] = 0
        
        if total_samples == 0:
            print("没有处理任何有效数据")
            return False
            
        print(f"\n总样本数: {total_samples}")
        
        # 2. 创建HDF5文件结构
        f = self._create_hdf5_structure(total_samples)
        
        try:
            # 3. 写入完整数据集
            current_idx = 0
            all_labels = []
            
            for class_data in all_data:
                num_samples = len(class_data['labels'])
                end_idx = current_idx + num_samples
                
                # 写入数据
                f['full']['stokes'][current_idx:end_idx] = class_data['stokes']
                f['full']['fluorescence'][current_idx:end_idx] = class_data['fluorescence']
                f['full']['images'][current_idx:end_idx] = class_data['images']
                f['full']['labels'][current_idx:end_idx] = class_data['labels']
                
                all_labels.extend(class_data['labels'])
                current_idx = end_idx
                
                # 释放内存
                del class_data
                self._check_memory()
                
            # 4. 创建分割索引
            labels = np.array(all_labels)
            indices = np.arange(len(labels))
            
            # 分层分割
            train_indices, temp_indices = train_test_split(
                indices,
                test_size=self.split_params['val_ratio'] + self.split_params['test_ratio'],
                stratify=labels,
                random_state=self.split_params['random_state']
            )
            
            val_indices, test_indices = train_test_split(
                temp_indices,
                test_size=self.split_params['test_ratio'] / (self.split_params['val_ratio'] + self.split_params['test_ratio']),
                stratify=labels[temp_indices],
                random_state=self.split_params['random_state']
            )
            
            # 保存分割索引
            split_indices = {
                'train': train_indices.tolist(),
                'val': val_indices.tolist(),
                'test': test_indices.tolist()
            }
            
            with open(self.output_path / 'split_indices.json', 'w') as sf:
                json.dump(split_indices, sf, indent=2)
            
            # 5. 写入分割数据集
            self._write_split_to_hdf5(f, 'train', train_indices)
            self._write_split_to_hdf5(f, 'val', val_indices)
            self._write_split_to_hdf5(f, 'test', test_indices)
            
            # 6. 生成兼容文件 (NPZ格式)
            print("\n生成兼容的NPZ文件...")
            for split_name in ['train', 'val', 'test']:
                split_data = {
                    'stokes': f[split_name]['stokes'][:],
                    'fluorescence': f[split_name]['fluorescence'][:],
                    'images': f[split_name]['images'][:],
                    'labels': f[split_name]['labels'][:]
                }
                
                npz_file = self.output_path / f'multimodal_data_{split_name}.npz'
                np.savez_compressed(npz_file, **split_data)
                print(f"{split_name}集已保存: {npz_file}")
                
                # 立即释放内存
                del split_data
                self._check_memory()
            
            # 7. 保存完整数据集
            full_data = {
                'stokes': f['full']['stokes'][:],
                'fluorescence': f['full']['fluorescence'][:],
                'images': f['full']['images'][:],
                'labels': f['full']['labels'][:]
            }
            
            np.savez_compressed(self.output_path / 'multimodal_data_full.npz', **full_data)
            np.savez_compressed(self.output_path / 'signal_data.npz', 
                              stokes=full_data['stokes'], 
                              fluorescence=full_data['fluorescence'], 
                              labels=full_data['labels'])
            np.savez_compressed(self.output_path / 'image_data.npz', 
                              images=full_data['images'], 
                              labels=full_data['labels'])
            
            del full_data
            self._check_memory()
            
            # 8. 保存元数据和统计信息
            self._save_metadata(total_samples, sample_counts)
            self._generate_metrics(f)
            
            print(f"\n处理完成!")
            print(f"HDF5文件: {self.hdf5_file}")
            print(f"总样本数: {total_samples}")
            
            return True
            
        finally:
            f.close()

def main():
    """主函数"""
    try:
        preprocessor = MultimodalHDF5Preprocessor()
        success = preprocessor.process()
        
        if success:
            print("多模态数据预处理完成!")
        else:
            print("预处理失败!")
            return 1
            
    except Exception as e:
        print(f"预处理过程出错: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
