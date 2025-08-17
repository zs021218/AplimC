from pathlib import Path
import numpy as np
import h5py
import json
from PIL import Image
import cv2
from typing import Dict, List, Tuple, Optional, Any
import yaml
from sklearn.model_selection import train_test_split
import gc
import psutil
import os
import re
import shutil

class MultimodalPolarFluDataPreprocessor:
    def __init__(self, raw_data_path, image_data_path, output_path):
        self.raw_data_path = Path(raw_data_path)
        self.image_data_path = Path(image_data_path)
        self.output_path = Path(output_path)
        
        # 确保输出目录存在
        self.output_path.mkdir(parents=True, exist_ok=True)

        # 定义类别映射
        self.class_map = {
            'CG': 0,
            'IG': 1,
            'PS3': 2,
            'PS6': 3,
            'PS10': 4,
            'QDDB': 5,
            'QZQG': 6,
            'SG': 7,
            'TP': 8,
            'TS': 9,
            'YMXH': 10,
            'YXXB': 11
        }

        # 反向映射
        self.inverse_class_map = {v: k for k, v in self.class_map.items()}
        
        # 数据结构参数
        self.signal_length = 4000
        self.rows_per_sample = 20
        self.num_views = 3  # 三视图
        
        # 图像处理参数
        self.image_size = (224, 224)  # 统一图像尺寸
        self.image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

        # 波长信息
        self.wavelengths = [445, 473, 520, 630]
        
        # 内存优化参数
        self.batch_size = 50  # 图像批处理大小
        self.use_float32 = True  # 强制使用float32节省内存
        self.memory_threshold = 0.85  # 内存使用阈值
        self.temp_dir = self.output_path / 'temp'
        self.temp_dir.mkdir(exist_ok=True)

    def get_class_name(self, label):
        """根据标签获取类别名称"""
        return self.inverse_class_map.get(label, f"Unknown_{label}")

    def get_class_label(self, class_name):
        """根据类别名称获取标签"""
        return self.class_map.get(class_name, -1)
    
    def check_memory_usage(self):
        """检查内存使用情况"""
        process = psutil.Process(os.getpid())
        memory_percent = process.memory_percent()
        if memory_percent > self.memory_threshold * 100:
            print(f"警告: 内存使用率达到 {memory_percent:.1f}%，触发垃圾回收")
            gc.collect()
            return True
        return False
        
    def load_mat_file(self, mat_path):
        """加载 MATLAB 文件（内存优化版本）"""
        try:
            print(f"加载文件: {mat_path}")
            with h5py.File(mat_path, 'r') as f:
                data_keys = [k for k in f.keys() if not k.startswith('#')]
                
                if not data_keys:
                    print("未找到有效数据键")
                    return None
                
                print(f"使用 h5py 读取，找到数据键: {data_keys}")
                
                # 读取数据并立即转换为指定精度
                data = f[data_keys[0]][()]
                if len(data.shape) == 2:
                    data = data.T
                
                # 转换为float32节省内存
                data = data.astype(np.float32)
                
                print(f"数据形状: {data.shape}, 数据类型: {data.dtype}")
                return data
                    
        except Exception as e:
            print(f"加载文件失败 {mat_path}: {e}")
            return None
            
    def extract_signal_samples(self, data):
        """从数据中提取信号样本（内存优化版本）"""
        if data is None:
            return None, None
            
        rows, cols = data.shape
        
        # 验证数据维度
        if cols != self.signal_length:
            print(f"警告: 信号长度不匹配。期望: {self.signal_length}, 实际: {cols}")

        # 样本数
        num_samples = rows // self.rows_per_sample
        print(f"提取信号样本数: {num_samples}")
        
        if num_samples == 0:
            print("警告: 数据行数不足以提取样本")
            return None, None
        
        # 使用float32初始化数组
        stokes_data = np.zeros((num_samples, 4, cols), dtype=np.float32)
        fluorescence_data = np.zeros((num_samples, 16, cols), dtype=np.float32)

        # 分批处理样本以节省内存
        batch_size = min(self.batch_size, num_samples)
        
        for batch_start in range(0, num_samples, batch_size):
            batch_end = min(batch_start + batch_size, num_samples)
            
            for sample_idx in range(batch_start, batch_end):
                start_row = sample_idx * self.rows_per_sample
                end_row = start_row + self.rows_per_sample
                
                if end_row > rows:
                    print(f"警告: 样本 {sample_idx} 数据不完整")
                    break
                    
                sample_data = data[start_row:end_row, :]

                # Stokes 参数 (前4行)
                stokes_data[sample_idx] = sample_data[0:4, :].astype(np.float32)

                # 荧光数据 (5-20行，共16个通道)
                fluorescence_data[sample_idx] = sample_data[4:20, :].astype(np.float32)
            
            # 批次处理完后检查内存
            if batch_end % 100 == 0:
                self.check_memory_usage()
                gc.collect()

        return stokes_data[:sample_idx+1], fluorescence_data[:sample_idx+1]

    def load_and_preprocess_image(self, image_path: Path) -> Optional[np.ndarray]:
        """加载并预处理单张图像（内存优化版本）"""
        try:
            # 使用PIL加载图像
            with Image.open(image_path) as image:
                # 转换为RGB（如果需要）
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # 调整尺寸
                image = image.resize(self.image_size, Image.Resampling.LANCZOS)
                
                # 转换为numpy数组并归一化到[0,1]
                image_array = np.array(image, dtype=np.float32) / 255.0
                
                return image_array
            
        except Exception as e:
            print(f"加载图像失败 {image_path}: {e}")
            return None

    def get_view_directories(self, class_name: str) -> List[Path]:
        """获取指定类别的视图目录"""
        # 构建类别图像目录路径
        class_image_dir = self.image_data_path / f"{class_name}-three views"
        
        if not class_image_dir.exists():
            print(f"警告: 未找到类别 {class_name} 的图像目录: {class_image_dir}")
            return []
        
        print(f"找到类别目录: {class_image_dir}")
        
        # 查找视图子目录
        view_dirs = []
        for view_num in range(1, self.num_views + 1):
            view_dir = class_image_dir / f"view{view_num}"
            if view_dir.exists():
                view_dirs.append(view_dir)
            else:
                print(f"警告: 未找到视图目录: {view_dir}")
        
        if len(view_dirs) != self.num_views:
            print(f"警告: 期望 {self.num_views} 个视图目录，实际找到 {len(view_dirs)} 个")
        
        return view_dirs

    def get_sample_images_from_views(self, view_dirs: List[Path]) -> List[List[Path]]:
        """从各个视图目录中获取样本图像"""
        if not view_dirs:
            return []
        
        # 获取每个视图目录中的图像文件
        view_images = {}
        for view_idx, view_dir in enumerate(view_dirs):
            images = []
            for ext in self.image_extensions:
                images.extend(view_dir.glob(f"*{ext}"))
                images.extend(view_dir.glob(f"*{ext.upper()}"))
            
            # 按文件名排序
            images = sorted(images, key=lambda x: self.extract_image_number(x.name))
            view_images[view_idx] = images
            print(f"视图 {view_idx + 1}: 找到 {len(images)} 张图像")
        
        # 确定最小样本数（取各视图中图像数的最小值）
        min_samples = min(len(images) for images in view_images.values()) if view_images else 0
        print(f"各视图最小样本数: {min_samples}")
        
        if min_samples == 0:
            return []
        
        # 组织样本：每个样本包含来自所有视图的对应图像
        samples = []
        for sample_idx in range(min_samples):
            sample_views = []
            for view_idx in range(len(view_dirs)):
                if sample_idx < len(view_images[view_idx]):
                    sample_views.append(view_images[view_idx][sample_idx])
                else:
                    print(f"警告: 视图 {view_idx + 1} 缺少样本 {sample_idx + 1}")
                    break
            
            # 只有当所有视图都有对应图像时才添加样本
            if len(sample_views) == len(view_dirs):
                samples.append(sample_views)
        
        print(f"成功组织 {len(samples)} 个完整样本")
        return samples

    def extract_image_number(self, filename: str) -> int:
        """从文件名中提取数字，用于排序"""
        import re
        numbers = re.findall(r'\d+', filename)
        return int(numbers[0]) if numbers else 0

    def process_class_images(self, class_name: str) -> Optional[np.ndarray]:
        """处理指定类别的所有图像（内存优化版本）"""
        print(f"\n处理 {class_name} 类别图像:")
        
        # 获取视图目录
        view_dirs = self.get_view_directories(class_name)
        if not view_dirs:
            return None
        
        # 获取组织好的样本图像
        sample_images = self.get_sample_images_from_views(view_dirs)
        if not sample_images:
            print(f"类别 {class_name} 没有有效的图像样本")
            return None
        
        num_samples = len(sample_images)
        
        # 预分配内存
        all_samples = np.zeros((num_samples, self.num_views, *self.image_size, 3), dtype=np.float32)
        
        # 分批处理图像以节省内存
        batch_size = min(self.batch_size, num_samples)
        processed_count = 0
        
        for batch_start in range(0, num_samples, batch_size):
            batch_end = min(batch_start + batch_size, num_samples)
            print(f"处理图像批次 {batch_start+1}-{batch_end}/{num_samples}")
            
            batch_processed = 0
            for sample_idx in range(batch_start, batch_end):
                sample_views = sample_images[sample_idx]
                views_loaded = 0
                
                for view_idx, image_path in enumerate(sample_views):
                    processed_image = self.load_and_preprocess_image(image_path)
                    if processed_image is not None:
                        all_samples[sample_idx, view_idx] = processed_image
                        views_loaded += 1
                    else:
                        print(f"警告: 无法处理图像 {image_path}")
                
                # 只有当所有视图都成功加载时才计数
                if views_loaded == len(sample_views):
                    batch_processed += 1
                    processed_count += 1
                else:
                    print(f"跳过样本 {sample_idx + 1}，部分视图加载失败")
            
            # 强制垃圾回收
            gc.collect()
            
            # 检查内存使用
            if self.check_memory_usage():
                print(f"内存使用过高，已触发垃圾回收")
        
        if processed_count == 0:
            print(f"类别 {class_name} 没有成功处理的图像样本")
            return None
        
        # 只返回成功处理的样本
        all_samples = all_samples[:processed_count]
        print(f"类别 {class_name} 图像数据形状: {all_samples.shape}")
        
        return all_samples

    def align_multimodal_data(self, stokes_data, fluorescence_data, image_data, class_name):
        """对齐多模态数据，确保样本数量一致"""
        signal_samples = len(stokes_data) if stokes_data is not None else 0
        image_samples = len(image_data) if image_data is not None else 0
        
        print(f"\n类别 {class_name} 数据对齐:")
        print(f"  信号样本数: {signal_samples}")
        print(f"  图像样本数: {image_samples}")
        
        if signal_samples == 0 or image_samples == 0:
            print(f"类别 {class_name} 缺少必要的模态数据")
            return None, None, None
        
        # 取最小样本数确保对齐
        aligned_samples = min(signal_samples, image_samples)
        print(f"  对齐后样本数: {aligned_samples}")
        
        if aligned_samples == 0:
            return None, None, None
        
        # 截取对齐的数据
        aligned_stokes = stokes_data[:aligned_samples]
        aligned_fluorescence = fluorescence_data[:aligned_samples]
        aligned_images = image_data[:aligned_samples]
        
        return aligned_stokes, aligned_fluorescence, aligned_images

    def _create_split_dataset(self, dataset, indices):
        """根据索引创建数据集分割（已弃用，使用内存优化版本）"""
        return {
            'stokes': dataset['stokes'][indices],
            'fluorescence': dataset['fluorescence'][indices],
            'images': dataset['images'][indices],
            'labels': dataset['labels'][indices],
            'indices': indices  # 保存原始索引用于调试
        }

    def _merge_class_data_files_streaming(self, class_data_files, sample_counts, save_splits, **split_kwargs):
        """流式合并类别数据文件 - 完全避免在内存中保存完整数据集"""
        total_samples = sum(count for _, _, count in class_data_files)
        
        print(f"准备流式合并 {len(class_data_files)} 个类别，总样本数: {total_samples}")
        
        # 先生成分割索引
        print("生成分割索引...")
        all_labels = []
        current_idx = 0
        
        # 收集所有标签以生成分层分割
        for temp_file, class_name, count in class_data_files:
            with np.load(temp_file) as data:
                all_labels.append(data['labels'])
        
        all_labels = np.concatenate(all_labels)
        indices = np.arange(total_samples)
        
        # 生成分割索引
        train_ratio = split_kwargs.get('train_ratio', 0.7)
        val_ratio = split_kwargs.get('val_ratio', 0.15)
        test_ratio = split_kwargs.get('test_ratio', 0.15)
        random_state = split_kwargs.get('random_state', 42)
        
        train_indices, temp_indices = train_test_split(
            indices, 
            test_size=(val_ratio + test_ratio),
            stratify=all_labels,
            random_state=random_state
        )
        
        val_indices, test_indices = train_test_split(
            temp_indices,
            test_size=test_ratio / (val_ratio + test_ratio),
            stratify=all_labels[temp_indices],
            random_state=random_state
        )
        
        # 保存分割索引
        split_indices = {
            'train': train_indices.tolist(),
            'val': val_indices.tolist(),
            'test': test_indices.tolist()
        }
        indices_path = self.output_path / 'split_indices.json'
        with open(indices_path, 'w') as f:
            json.dump(split_indices, f, indent=2)
        print(f"分割索引已保存到: {indices_path}")
        
        # 释放标签内存
        del all_labels
        gc.collect()
        
        if save_splits:
            # 流式创建分割数据集
            self._create_splits_streaming(class_data_files, 
                                        {'train': train_indices, 'val': val_indices, 'test': test_indices})
            
            # 生成统计信息
            splits_info = self._generate_splits_info(class_data_files, 
                                                   {'train': train_indices, 'val': val_indices, 'test': test_indices})
            
            # 清理临时文件
            self._cleanup_temp_files(class_data_files)
            
            # 创建简化的数据集信息
            dataset_info = {
                'class_map': self.class_map,
                'inverse_class_map': self.inverse_class_map,
                'sample_counts': sample_counts,
                'wavelengths': self.wavelengths,
                'signal_length': self.signal_length,
                'image_size': self.image_size,
                'num_views': self.num_views,
                'total_samples': total_samples
            }
            
            return dataset_info, splits_info
        
        return None

    def process_class_data(self, class_name):
        """处理特定类别的多模态数据（内存优化版本）"""
        print(f"\n{'='*60}")
        print(f"处理 {class_name} 类别多模态数据")
        print(f"{'='*60}")

        # 1. 处理信号数据
        mat_file = self.raw_data_path / f"{class_name}.mat"
        if not mat_file.exists():
            print(f"未找到 {class_name} 类别的 .mat 文件: {mat_file}")
            return None

        print(f"处理信号文件: {mat_file.name}")
        signal_data = self.load_mat_file(mat_file)
        if signal_data is None:
            print(f"无法加载信号文件: {mat_file.name}")
            return None

        stokes_data, fluorescence_data = self.extract_signal_samples(signal_data)
        
        # 立即释放原始信号数据内存
        del signal_data
        gc.collect()
        
        if stokes_data is None or fluorescence_data is None:
            print(f"无法从 {mat_file.name} 提取信号样本")
            return None

        # 2. 处理图像数据
        image_data = self.process_class_images(class_name)
        
        if image_data is None:
            print(f"类别 {class_name} 图像处理失败")
            return None
        
        # 3. 数据对齐
        aligned_stokes, aligned_fluorescence, aligned_images = self.align_multimodal_data(
            stokes_data, fluorescence_data, image_data, class_name
        )
        
        # 释放未对齐的数据
        del stokes_data, fluorescence_data, image_data
        gc.collect()
        
        if aligned_stokes is None:
            print(f"类别 {class_name} 数据对齐失败")
            return None

        # 4. 创建标签
        class_label = self.get_class_label(class_name)
        if class_label == -1:
            print(f"警告: 未知类别 {class_name}")
            return None
            
        num_samples = len(aligned_stokes)
        labels = np.full(num_samples, class_label, dtype=np.int32)

        print(f"\n类别 {class_name} 处理结果:")
        print(f"  类别标签: {class_label}")
        print(f"  样本数量: {num_samples}")
        print(f"  Stokes数据形状: {aligned_stokes.shape}")
        print(f"  荧光数据形状: {aligned_fluorescence.shape}")
        print(f"  图像数据形状: {aligned_images.shape}")
        
        # 最后检查内存使用
        self.check_memory_usage()

        return {
            'stokes': aligned_stokes,
            'fluorescence': aligned_fluorescence,
            'images': aligned_images,
            'labels': labels,
            'class_name': class_name
        }
        
    def preprocess_all_classes_memory_efficient(self, save_splits=True, **split_kwargs):
        """内存高效的全类别预处理 - 分类别处理避免内存溢出"""
        print(f"开始处理 {len(self.class_map)} 个类别的多模态数据（内存优化版本）...")

        # 分别保存每个类别的数据，避免同时加载所有数据
        class_data_files = []
        sample_counts = {}

        for class_name in self.class_map.keys():
            print(f"\n处理类别: {class_name}")
            
            class_data = self.process_class_data(class_name)
            if class_data is None:
                print(f"跳过类别 {class_name}")
                sample_counts[class_name] = 0
                continue
            
            # 保存单个类别数据到临时文件
            temp_file = self.temp_dir / f"temp_{class_name}.npz"
            np.savez_compressed(temp_file, **class_data)
            class_data_files.append((temp_file, class_name, len(class_data['labels'])))
            sample_counts[class_name] = len(class_data['labels'])
            
            # 释放内存
            del class_data
            gc.collect()
            
            print(f"类别 {class_name} 数据已保存到临时文件，释放内存")

        if not class_data_files:
            print("没有成功处理的类别数据")
            return None

        # 现在合并所有数据
        print("\n流式合并所有类别数据...")
        return self._merge_class_data_files_streaming(class_data_files, sample_counts, save_splits, **split_kwargs)

    def _create_splits_streaming(self, class_data_files, split_indices_dict):
        """流式创建分割数据集（修复版本）"""
        print("\n开始流式创建分割数据集（内存优化版本）...")
        
        # 创建分割索引集合以快速查找
        split_index_sets = {
            split_name: set(indices) 
            for split_name, indices in split_indices_dict.items()
        }
        
        # 初始化每个分割的临时文件列表
        split_temp_files = {split_name: [] for split_name in split_indices_dict.keys()}
        
        current_global_idx = 0
        
        # 逐个处理类别数据
        for temp_file, class_name, count in class_data_files:
            print(f"流式处理类别 {class_name} 数据...")
            
            # 加载类别数据
            with np.load(temp_file) as data:
                class_stokes = data['stokes']
                class_fluorescence = data['fluorescence']
                class_images = data['images']
                class_labels = data['labels']
                
                # 初始化每个分割的缓冲区
                split_buffers = {
                    split_name: {
                        'stokes': [],
                        'fluorescence': [],
                        'images': [],
                        'labels': []
                    } for split_name in split_indices_dict.keys()
                }
                
                # 分配每个样本到对应的分割
                for local_idx in range(count):
                    global_idx = current_global_idx + local_idx
                    
                    # 查找该样本属于哪个分割
                    for split_name, index_set in split_index_sets.items():
                        if global_idx in index_set:
                            split_buffers[split_name]['stokes'].append(class_stokes[local_idx])
                            split_buffers[split_name]['fluorescence'].append(class_fluorescence[local_idx])
                            split_buffers[split_name]['images'].append(class_images[local_idx])
                            split_buffers[split_name]['labels'].append(class_labels[local_idx])
                            break
                
                # 保存每个分割的缓冲区到临时文件
                for split_name, buffer in split_buffers.items():
                    if buffer['stokes']:  # 如果有数据
                        temp_split_file = self.temp_dir / f"temp_split_{split_name}_{class_name}.npz"
                        np.savez_compressed(temp_split_file,
                                          stokes=np.array(buffer['stokes'], dtype=np.float32),
                                          fluorescence=np.array(buffer['fluorescence'], dtype=np.float32),
                                          images=np.array(buffer['images'], dtype=np.float32),
                                          labels=np.array(buffer['labels'], dtype=np.int32))
                        split_temp_files[split_name].append(temp_split_file)
                        print(f"  {split_name}集: {len(buffer['stokes'])} 样本已缓存")
                
                current_global_idx += count
            
            print(f"类别 {class_name} 处理完成，内存已释放")
            gc.collect()
        
        # 合并每个分割的临时文件
        for split_name, temp_files in split_temp_files.items():
            if temp_files:
                self._merge_split_temp_files_final(split_name, temp_files)
            else:
                print(f"警告: {split_name}集没有数据!")
        
        print("流式分割完成")

    def _merge_split_temp_files_final(self, split_name, temp_files):
        """最终合并分割的临时文件"""
        print(f"合并{split_name}集的{len(temp_files)}个临时文件...")
        
        # 计算总样本数
        total_samples = 0
        for temp_file in temp_files:
            with np.load(temp_file) as data:
                total_samples += len(data['labels'])
        
        if total_samples == 0:
            print(f"警告: {split_name}集没有样本!")
            return
        
        print(f"{split_name}集总样本数: {total_samples}")
        
        # 预分配最终数组
        final_stokes = np.zeros((total_samples, 4, self.signal_length), dtype=np.float32)
        final_fluorescence = np.zeros((total_samples, 16, self.signal_length), dtype=np.float32)
        final_images = np.zeros((total_samples, self.num_views, *self.image_size, 3), dtype=np.float32)
        final_labels = np.zeros(total_samples, dtype=np.int32)
        
        current_idx = 0
        
        # 合并所有临时文件
        for temp_file in temp_files:
            with np.load(temp_file) as data:
                batch_size = len(data['labels'])
                end_idx = current_idx + batch_size
                
                final_stokes[current_idx:end_idx] = data['stokes']
                final_fluorescence[current_idx:end_idx] = data['fluorescence']
                final_images[current_idx:end_idx] = data['images']
                final_labels[current_idx:end_idx] = data['labels']
                
                current_idx = end_idx
            
            # 删除临时文件
            temp_file.unlink()
        
        # 保存最终文件
        split_path = self.output_path / f'multimodal_data_{split_name}.npz'
        np.savez_compressed(split_path,
                          stokes=final_stokes,
                          fluorescence=final_fluorescence,
                          images=final_images,
                          labels=final_labels)
        
        print(f"{split_name}集已保存到: {split_path} ({total_samples} 样本)")
        
        # 保存元数据
        self._save_split_metadata(split_name, final_stokes.shape, final_fluorescence.shape, 
                                final_images.shape, final_labels.shape)
        
        # 清理内存
        del final_stokes, final_fluorescence, final_images, final_labels
        gc.collect()

    def _flush_split_buffer(self, split_name, buffer, final=False):
        """清空分割缓冲区到文件"""
        if not buffer['stokes']:
            return
            
        split_path = self.output_path / f'multimodal_data_{split_name}.npz'
        
        # 转换为numpy数组
        stokes_batch = np.array(buffer['stokes'], dtype=np.float32)
        fluorescence_batch = np.array(buffer['fluorescence'], dtype=np.float32)
        images_batch = np.array(buffer['images'], dtype=np.float32)
        labels_batch = np.array(buffer['labels'], dtype=np.int32)
        
        if split_path.exists() and not final:
            # 追加到现有文件 - 实际上numpy不支持追加，所以我们需要重新实现
            # 这里我们暂时保存到临时文件，最后合并
            temp_path = self.temp_dir / f"temp_split_{split_name}_{len(buffer['stokes'])}.npz"
            np.savez_compressed(temp_path,
                               stokes=stokes_batch,
                               fluorescence=fluorescence_batch,
                               images=images_batch,
                               labels=labels_batch)
        else:
            # 最终保存
            if final:
                # 如果有临时分割文件，需要合并
                temp_files = list(self.temp_dir.glob(f"temp_split_{split_name}_*.npz"))
                if temp_files:
                    self._merge_split_temp_files(split_name, temp_files, 
                                               stokes_batch, fluorescence_batch, images_batch, labels_batch)
                else:
                    # 直接保存
                    np.savez_compressed(split_path,
                                       stokes=stokes_batch,
                                       fluorescence=fluorescence_batch,
                                       images=images_batch,
                                       labels=labels_batch)
                    
                    print(f"{split_name}集已保存到: {split_path} ({len(labels_batch)} 样本)")
                    
                    # 保存元数据
                    self._save_split_metadata(split_name, stokes_batch.shape, fluorescence_batch.shape, 
                                            images_batch.shape, labels_batch.shape)
        
        # 清空缓冲区
        buffer['stokes'].clear()
        buffer['fluorescence'].clear() 
        buffer['images'].clear()
        buffer['labels'].clear()
        
        # 释放内存
        del stokes_batch, fluorescence_batch, images_batch, labels_batch
        gc.collect()

    def _merge_split_temp_files(self, split_name, temp_files, final_stokes, final_fluorescence, final_images, final_labels):
        """合并分割临时文件"""
        print(f"合并 {split_name} 集的临时文件...")
        
        all_stokes = [final_stokes] if final_stokes.size > 0 else []
        all_fluorescence = [final_fluorescence] if final_fluorescence.size > 0 else []
        all_images = [final_images] if final_images.size > 0 else []
        all_labels = [final_labels] if final_labels.size > 0 else []
        
        for temp_file in temp_files:
            with np.load(temp_file) as data:
                all_stokes.append(data['stokes'])
                all_fluorescence.append(data['fluorescence'])
                all_images.append(data['images'])
                all_labels.append(data['labels'])
            temp_file.unlink()  # 删除临时文件
        
        if all_stokes:
            merged_stokes = np.vstack(all_stokes)
            merged_fluorescence = np.vstack(all_fluorescence)
            merged_images = np.vstack(all_images)
            merged_labels = np.concatenate(all_labels)
            
            split_path = self.output_path / f'multimodal_data_{split_name}.npz'
            np.savez_compressed(split_path,
                               stokes=merged_stokes,
                               fluorescence=merged_fluorescence,
                               images=merged_images,
                               labels=merged_labels)
            
            print(f"{split_name}集已保存到: {split_path} ({len(merged_labels)} 样本)")
            
            # 保存元数据
            self._save_split_metadata(split_name, merged_stokes.shape, merged_fluorescence.shape,
                                    merged_images.shape, merged_labels.shape)
            
            # 释放内存
            del merged_stokes, merged_fluorescence, merged_images, merged_labels
            
        gc.collect()

    def _save_split_metadata(self, split_name, stokes_shape, fluorescence_shape, images_shape, labels_shape):
        """保存分割元数据"""
        split_metadata = {
            'split': split_name,
            'total_samples': labels_shape[0],
            'data_shapes': {
                'stokes': list(stokes_shape),
                'fluorescence': list(fluorescence_shape),
                'images': list(images_shape),
                'labels': list(labels_shape)
            }
        }
        
        metadata_path = self.output_path / f'metadata_{split_name}.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(split_metadata, f, indent=2, ensure_ascii=False)

    def _generate_splits_info(self, class_data_files, split_indices_dict):
        """为统计生成分割信息"""
        print("\n生成分割统计信息...")
        
        # 重新读取标签信息
        all_labels = []
        for temp_file, class_name, count in class_data_files:
            if temp_file.exists():  # 检查文件是否还存在
                with np.load(temp_file) as data:
                    all_labels.append(data['labels'])
        
        if not all_labels:
            return {}
            
        all_labels = np.concatenate(all_labels)
        
        splits_info = {}
        for split_name, indices in split_indices_dict.items():
            split_labels = all_labels[indices]
            splits_info[split_name] = {
                'total_samples': len(split_labels),
                'labels': split_labels
            }
        
        return splits_info

    def _cleanup_temp_files(self, class_data_files):
        """清理临时文件"""
        print("\n清理临时文件...")
        for temp_file, class_name, count in class_data_files:
            if temp_file.exists():
                temp_file.unlink()
        
        # 删除临时目录
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
            self.temp_dir.mkdir(exist_ok=True)
        except:
            pass

    def split_dataset_memory_efficient(self, dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
        """内存高效的数据集分割 - 直接保存避免创建多份副本"""
        print(f"\n{'='*50}")
        print("内存高效数据集分割")
        print(f"{'='*50}")
        
        # 验证比例
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须等于1"
        
        total_samples = len(dataset['labels'])
        labels = dataset['labels']
        indices = np.arange(total_samples)
        
        print(f"总样本数: {total_samples}")
        
        # 第一次分割: 训练集 vs (验证集+测试集)
        train_indices, temp_indices = train_test_split(
            indices, 
            test_size=(val_ratio + test_ratio),
            stratify=labels,
            random_state=random_state
        )
        
        # 第二次分割: 验证集 vs 测试集
        val_indices, test_indices = train_test_split(
            temp_indices,
            test_size=test_ratio / (val_ratio + test_ratio),
            stratify=labels[temp_indices],
            random_state=random_state
        )
        
        # 保存分割索引
        split_indices = {
            'train': train_indices.tolist(),
            'val': val_indices.tolist(),
            'test': test_indices.tolist()
        }
        indices_path = self.output_path / 'split_indices.json'
        with open(indices_path, 'w') as f:
            json.dump(split_indices, f, indent=2)
        print(f"分割索引已保存到: {indices_path}")
        
        # 直接保存各个分割，不在内存中创建副本
        splits_info = {}
        for split_name, split_indices in [('train', train_indices), ('val', val_indices), ('test', test_indices)]:
            print(f"\n保存 {split_name} 集 ({len(split_indices)} 样本)...")
            
            # 直接保存分割数据，避免创建内存副本
            split_path = self.output_path / f'multimodal_data_{split_name}.npz'
            
            # 直接使用索引切片，避免分批累积 - 这是内存爆炸的根本原因
            print(f"直接切片 {split_name} 数据 ({len(split_indices)} 样本)...")
            
            # 直接索引切片，不分批不累积
            split_stokes = dataset['stokes'][split_indices]
            split_fluorescence = dataset['fluorescence'][split_indices]
            split_images = dataset['images'][split_indices]
            split_labels = dataset['labels'][split_indices]
            
            print(f"保存 {split_name} 数据到文件...")
            # 立即保存
            np.savez_compressed(split_path, 
                               stokes=split_stokes,
                               fluorescence=split_fluorescence, 
                               images=split_images,
                               labels=split_labels)
            
            print(f"{split_name}集已保存到: {split_path}")
            
            # 保存分割元数据
            split_metadata = {
                'split': split_name,
                'total_samples': len(split_labels),
                'data_shapes': {
                    'stokes': list(split_stokes.shape),
                    'fluorescence': list(split_fluorescence.shape),
                    'images': list(split_images.shape),
                    'labels': list(split_labels.shape)
                }
            }
            
            metadata_path = self.output_path / f'metadata_{split_name}.json'
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(split_metadata, f, indent=2, ensure_ascii=False)
            
            # 收集统计信息
            splits_info[split_name] = {
                'total_samples': len(split_labels),
                'labels': split_labels.copy()  # 复制一份用于统计
            }
            
            # 立即释放分割数据内存 - 关键！
            del split_stokes, split_fluorescence, split_images, split_labels
            gc.collect()
            
            print(f"{split_name}集已保存并释放内存")
            
            # 强制内存检查
            self.check_memory_usage()
        
        # 打印统计信息
        self._print_split_statistics_from_info(splits_info, dataset['class_map'])
        
        return splits_info
    
    def _create_split_dataset(self, dataset, indices):
        """根据索引创建数据集分割"""
        return {
            'stokes': dataset['stokes'][indices],
            'fluorescence': dataset['fluorescence'][indices],
            'images': dataset['images'][indices],
            'labels': dataset['labels'][indices],
            'indices': indices  # 保存原始索引用于调试
        }
    
    def _print_split_statistics(self, splits, class_map):
        """打印数据集分割统计信息"""
        print(f"\n数据集分割统计:")
        print(f"{'='*60}")
        
        for split_name, split_data in splits.items():
            labels = split_data['labels']
            total = len(labels)
            
            print(f"\n{split_name.upper()}集: {total} 样本")
            print("-" * 30)
            
            # 统计各类别样本数
            for class_name, class_label in class_map.items():
                count = np.sum(labels == class_label)
                percentage = count / total * 100 if total > 0 else 0
                print(f"  {class_name}: {count} 样本 ({percentage:.1f}%)")
    
    def _print_split_statistics_from_info(self, splits_info, class_map):
        """从分割信息打印统计"""
        print(f"\n数据集分割统计:")
        print(f"{'='*60}")
        
        for split_name, split_info in splits_info.items():
            labels = split_info['labels']
            total = len(labels)
            
            print(f"\n{split_name.upper()}集: {total} 样本")
            print("-" * 30)
            
            # 统计各类别样本数
            for class_name, class_label in class_map.items():
                count = np.sum(labels == class_label)
                percentage = count / total * 100 if total > 0 else 0
                print(f"  {class_name}: {count} 样本 ({percentage:.1f}%)")
    
    def save_split_datasets(self, splits, base_filename='multimodal_data'):
        """保存分割后的数据集"""
        print(f"\n{'='*50}")
        print("保存分割数据集")
        print(f"{'='*50}")
        
        for split_name, split_data in splits.items():
            # 保存完整分割数据
            split_path = self.output_path / f'{base_filename}_{split_name}.npz'
            
            # 移除索引信息（仅用于内部）
            save_data = {k: v for k, v in split_data.items() if k != 'indices'}
            
            np.savez_compressed(split_path, **save_data)
            print(f"{split_name}集已保存到: {split_path}")
            
            # 保存分割的元数据
            split_metadata = {
                'split': split_name,
                'total_samples': len(split_data['labels']),
                'data_shapes': {
                    'stokes': list(split_data['stokes'].shape),
                    'fluorescence': list(split_data['fluorescence'].shape),
                    'images': list(split_data['images'].shape),
                    'labels': list(split_data['labels'].shape)
                }
            }
            
            metadata_path = self.output_path / f'metadata_{split_name}.json'
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(split_metadata, f, indent=2, ensure_ascii=False)
        
        # 保存分割索引信息（用于重现性和调试）
        split_indices = {
            split_name: split_data['indices'].tolist() 
            for split_name, split_data in splits.items()
        }
        
        indices_path = self.output_path / 'split_indices.json'
        with open(indices_path, 'w') as f:
            json.dump(split_indices, f, indent=2)
        print(f"分割索引已保存到: {indices_path}")

    def validate_multimodal_consistency(self, stokes_data, fluorescence_data, image_data, labels):
        """验证多模态数据的一致性"""
        print(f"\n{'='*40}")
        print("多模态数据一致性验证")
        print(f"{'='*40}")
        
        # 检查样本数量一致性
        num_stokes = len(stokes_data)
        num_fluorescence = len(fluorescence_data)
        num_images = len(image_data)
        num_labels = len(labels)
        
        print(f"Stokes样本数: {num_stokes}")
        print(f"荧光样本数: {num_fluorescence}")
        print(f"图像样本数: {num_images}")
        print(f"标签数: {num_labels}")
        
        if not (num_stokes == num_fluorescence == num_images == num_labels):
            raise ValueError(f"多模态数据样本数不一致!")
        
        print("✓ 多模态数据样本数一致性验证通过")
        
        # 检查标签有效性
        self.validate_labels(labels)
        
        return True
        
    def validate_labels(self, labels):
        """验证标签的有效性"""
        print(f"\n{'='*30}")
        print("标签验证")
        print(f"{'='*30}")
        
        unique_labels = np.unique(labels)
        valid_labels = set(self.class_map.values())
        
        print(f"发现的唯一标签: {sorted(unique_labels)}")
        print(f"有效标签范围: {sorted(valid_labels)}")
        
        # 检查无效标签
        invalid_labels = set(unique_labels) - valid_labels
        if invalid_labels:
            print(f"警告: 发现无效标签: {sorted(invalid_labels)}")
            return False
        
        # 检查缺失标签
        missing_labels = valid_labels - set(unique_labels)
        if missing_labels:
            print(f"注意: 缺失标签: {sorted(missing_labels)}")
            missing_classes = [self.get_class_name(label) for label in missing_labels]
            print(f"对应类别: {missing_classes}")
        
        # 统计每个类别的样本数
        print(f"\n各类别样本统计:")
        for label in sorted(unique_labels):
            count = np.sum(labels == label)
            class_name = self.get_class_name(label)
            print(f"  {class_name} (标签{label}): {count} 个样本")
        
        print(f"✓ 标签验证完成")
        return True

    def get_label_statistics(self, labels):
        """获取标签统计信息"""
        stats = {}
        unique_labels = np.unique(labels)
        total_samples = len(labels)
        
        for label in unique_labels:
            count = np.sum(labels == label)
            class_name = self.get_class_name(label)
            stats[class_name] = {
                'label': int(label),
                'count': int(count),
                'percentage': float(count / total_samples * 100)
            }
        
        return stats
        
    def save_dataset(self, dataset, filename='multimodal_data.npz'):
        """保存处理后的多模态数据集"""
        save_path = self.output_path / filename
        
        try:
            # 分别保存不同类型的数据
            signal_data = {
                'stokes': dataset['stokes'],
                'fluorescence': dataset['fluorescence'],
                'labels': dataset['labels']
            }
            
            # 保存信号数据
            signal_path = self.output_path / 'signal_data.npz'
            np.savez_compressed(signal_path, **signal_data)
            print(f"信号数据已保存到: {signal_path}")
            
            # 保存图像数据
            image_path = self.output_path / 'image_data.npz'
            np.savez_compressed(image_path, images=dataset['images'], labels=dataset['labels'])
            print(f"图像数据已保存到: {image_path}")
            
            # 保存完整数据集（可能很大）
            np.savez_compressed(save_path, **dataset)
            print(f"完整数据集已保存到: {save_path}")
            
            # 保存元数据
            metadata = {
                'class_map': dataset['class_map'],
                'sample_counts': dataset['sample_counts'],
                'wavelengths': dataset['wavelengths'],
                'signal_length': dataset['signal_length'],
                'image_size': list(dataset['image_size']),
                'num_views': dataset['num_views'],
                'total_samples': int(len(dataset['labels'])),
                'data_shapes': {
                    'stokes': list(dataset['stokes'].shape),
                    'fluorescence': list(dataset['fluorescence'].shape),
                    'images': list(dataset['images'].shape),
                    'labels': list(dataset['labels'].shape)
                }
            }
            
            metadata_path = self.output_path / 'metadata.json'
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            print(f"元数据已保存到: {metadata_path}")
            
        except Exception as e:
            print(f"保存失败: {e}")

    def load_dataset(self, filename='multimodal_data.npz'):
        """加载处理后的多模态数据集"""
        load_path = self.output_path / filename
        if not load_path.exists():
            print(f"文件不存在: {load_path}")
            return None
        
        try:
            data = np.load(load_path, allow_pickle=True)
            dataset = {key: data[key] for key in data.files}
            print(f"多模态数据集已从 {load_path} 加载")
            return dataset
        except Exception as e:
            print(f"加载失败: {e}")
            return None

    @staticmethod
    def load_params(params_file='params.yaml'):
        """加载参数文件"""
        try:
            with open(params_file, 'r') as f:
                params = yaml.safe_load(f)
            return params
        except Exception as e:
            print(f"加载参数文件失败: {e}")
            return None

    def generate_split_metrics(self, splits, class_map, metrics_path=None):
        """生成分割统计信息并保存为JSON（用于DVC metrics）"""
        metrics = {}
        for split_name, split_data in splits.items():
            labels = split_data['labels']
            total = len(labels)
            split_stats = {}
            for class_name, class_label in class_map.items():
                count = int(np.sum(labels == class_label))
                percentage = float(count / total * 100) if total > 0 else 0.0
                split_stats[class_name] = {
                    'label': class_label,
                    'count': count,
                    'percentage': percentage
                }
            metrics[split_name] = {
                'total_samples': total,
                'class_stats': split_stats
            }
        # 保存到JSON文件（可选）
        if metrics_path is None:
            metrics_path = self.output_path / 'split_metrics.json'
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"分割统计信息已保存到: {metrics_path}")
        return metrics

    def generate_split_metrics_from_info(self, splits_info, class_map, metrics_path=None):
        """从分割信息生成统计信息并保存为JSON（用于DVC metrics）"""
        metrics = {}
        for split_name, split_info in splits_info.items():
            labels = split_info['labels']
            total = len(labels)
            split_stats = {}
            for class_name, class_label in class_map.items():
                count = int(np.sum(labels == class_label))
                percentage = float(count / total * 100) if total > 0 else 0.0
                split_stats[class_name] = {
                    'label': class_label,
                    'count': count,
                    'percentage': percentage
                }
            metrics[split_name] = {
                'total_samples': total,
                'class_stats': split_stats
            }
        # 保存到JSON文件
        if metrics_path is None:
            metrics_path = self.output_path / 'split_metrics.json'
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"分割统计信息已保存到: {metrics_path}")
        return metrics

def main():
    # 加载参数
    params = MultimodalPolarFluDataPreprocessor.load_params()
    if params:
        # 从参数文件获取配置
        raw_data_path = Path(params['paths']['raw_data'])
        image_data_path = Path(params['paths']['image_data'])
        output_path = Path(params['paths']['output'])
        
        # 分割参数
        split_params = params.get('split', {})
        train_ratio = split_params.get('train_ratio', 0.7)
        val_ratio = split_params.get('val_ratio', 0.15)
        test_ratio = split_params.get('test_ratio', 0.15)
        random_state = split_params.get('random_state', 42)
    else:
        # 使用默认参数
        raw_data_path = Path('/data3/zs/AplimC/data/raw/polar_flu_data')
        image_data_path = Path('/data3/zs/AplimC/data/raw/images')
        output_path = Path('/data3/zs/AplimC/data/raw/processed_data')
        train_ratio, val_ratio, test_ratio, random_state = 0.7, 0.15, 0.15, 42

    # 创建多模态预处理器实例
    preprocessor = MultimodalPolarFluDataPreprocessor(
        raw_data_path=raw_data_path,
        image_data_path=image_data_path,
        output_path=output_path
    )

    # 如果有参数，更新配置
    if params:
        preprocess_params = params.get('preprocess', {})
        preprocessor.signal_length = preprocess_params.get('signal_length', 4000)
        preprocessor.rows_per_sample = preprocess_params.get('rows_per_sample', 20)
        preprocessor.image_size = tuple(preprocess_params.get('image_size', [224, 224]))
        preprocessor.num_views = preprocess_params.get('num_views', 3)
        preprocessor.wavelengths = preprocess_params.get('wavelengths', [445, 473, 520, 630])
        # 更新内存优化参数
        preprocessor.batch_size = preprocess_params.get('batch_size', 50)
        preprocessor.use_float32 = preprocess_params.get('use_float32', True)
        preprocessor.memory_threshold = preprocess_params.get('memory_threshold', 0.85)

    # 处理所有类别数据并分割
    print("开始多模态数据预处理（内存优化版本）...")
    result = preprocessor.preprocess_all_classes_memory_efficient(
        save_splits=True,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=random_state
    )
    
    # 生成分割统计信息（用于DVC metrics）
    if result is not None and isinstance(result, tuple):
        dataset_info, splits_info = result
        preprocessor.generate_split_metrics_from_info(splits_info, dataset_info['class_map'])
    
if __name__ == "__main__":
    main()
    print("多模态数据预处理完成。")