from pathlib import Path
import numpy as np
import h5py
import json
from PIL import Image
import cv2
from typing import Dict, List, Tuple, Optional, Any
import yaml

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

    def get_class_name(self, label):
        """根据标签获取类别名称"""
        return self.inverse_class_map.get(label, f"Unknown_{label}")

    def get_class_label(self, class_name):
        """根据类别名称获取标签"""
        return self.class_map.get(class_name, -1)
        
    def load_mat_file(self, mat_path):
        """加载 MATLAB 文件"""
        try:
            with h5py.File(mat_path, 'r') as f:
                data_keys = [k for k in f.keys() if not k.startswith('#')]
                mat_data = {}
                
                for key in data_keys:
                    if isinstance(f[key], h5py.Dataset):
                        data = f[key][()]
                        if len(data.shape) == 2:
                            data = data.T
                        mat_data[key] = data
                
                print(f"使用 h5py 读取，找到数据键: {data_keys}")
                
                if data_keys:
                    return mat_data[data_keys[0]]
                else:
                    print("未找到有效数据键")
                    return None
                    
        except Exception as e:
            print(f"加载文件失败 {mat_path}: {e}")
            return None
            
    def extract_signal_samples(self, data):
        """从数据中提取信号样本"""
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
        
        # 初始化数组 
        stokes_data = np.zeros((num_samples, 4, cols))
        fluorescence_data = np.zeros((num_samples, 16, cols))

        for sample_idx in range(num_samples):
            start_row = sample_idx * self.rows_per_sample
            end_row = start_row + self.rows_per_sample
            
            if end_row > rows:
                print(f"警告: 样本 {sample_idx} 数据不完整")
                break
                
            sample_data = data[start_row:end_row, :]

            # Stokes 参数 (前4行)
            stokes_data[sample_idx] = sample_data[0:4, :]

            # 荧光数据 (5-20行，共16个通道)
            fluorescence_data[sample_idx] = sample_data[4:20, :]

        return stokes_data[:sample_idx+1], fluorescence_data[:sample_idx+1]

    def load_and_preprocess_image(self, image_path: Path) -> Optional[np.ndarray]:
        """加载并预处理单张图像"""
        try:
            # 使用PIL加载图像
            image = Image.open(image_path).convert('RGB')
            
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
        """处理指定类别的所有图像"""
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
        
        # 预处理所有图像样本
        processed_samples = []
        for sample_idx, sample_views in enumerate(sample_images):
            processed_views = []
            
            for view_idx, image_path in enumerate(sample_views):
                processed_image = self.load_and_preprocess_image(image_path)
                if processed_image is None:
                    print(f"跳过样本 {sample_idx + 1}，视图 {view_idx + 1} 加载失败: {image_path}")
                    break
                processed_views.append(processed_image)
            
            # 确保所有视图都成功加载
            if len(processed_views) == len(sample_views):
                # 堆叠为 (num_views, H, W, C)
                sample_array = np.stack(processed_views, axis=0)
                processed_samples.append(sample_array)
            else:
                print(f"跳过样本 {sample_idx + 1}，部分视图加载失败")
        
        if not processed_samples:
            print(f"类别 {class_name} 没有成功处理的图像样本")
            return None
        
        # 合并所有样本: (num_samples, num_views, H, W, C)
        all_samples = np.stack(processed_samples, axis=0)
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

    def process_class_data(self, class_name):
        """处理特定类别的多模态数据"""
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
        if stokes_data is None or fluorescence_data is None:
            print(f"无法从 {mat_file.name} 提取信号样本")
            return None

        # 2. 处理图像数据
        image_data = self.process_class_images(class_name)
        
        # 3. 数据对齐
        aligned_stokes, aligned_fluorescence, aligned_images = self.align_multimodal_data(
            stokes_data, fluorescence_data, image_data, class_name
        )
        
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

        return {
            'stokes': aligned_stokes,
            'fluorescence': aligned_fluorescence,
            'images': aligned_images,
            'labels': labels,
            'class_name': class_name
        }
        
    def preprocess_all_classes(self):
        """预处理所有类别的多模态数据"""
        all_stokes = []
        all_fluorescence = []
        all_images = []
        all_labels = []
        sample_counts = {}

        print(f"开始处理 {len(self.class_map)} 个类别的多模态数据...")

        for class_name in self.class_map.keys():
            result = self.process_class_data(class_name)
            if result is None:
                sample_counts[class_name] = 0
                continue
                
            all_stokes.append(result['stokes'])
            all_fluorescence.append(result['fluorescence'])
            all_images.append(result['images'])
            all_labels.append(result['labels'])
            sample_counts[class_name] = len(result['labels'])

        if not all_stokes:
            print("错误: 没有处理任何数据")
            return None

        # 合并所有类别的数据
        try:
            all_stokes = np.vstack(all_stokes)
            all_fluorescence = np.vstack(all_fluorescence)
            all_images = np.vstack(all_images)
            all_labels = np.concatenate(all_labels)
        except Exception as e:
            print(f"数据合并失败: {e}")
            return None

        print(f"\n{'='*60}")
        print(f"多模态数据合并完成")
        print(f"{'='*60}")
        print(f"总样本数: {len(all_stokes)}")
        print(f"Stokes数据形状: {all_stokes.shape}")
        print(f"荧光数据形状: {all_fluorescence.shape}")
        print(f"图像数据形状: {all_images.shape}")
        print(f"标签形状: {all_labels.shape}")

        # 验证多模态数据一致性
        self.validate_multimodal_consistency(all_stokes, all_fluorescence, all_images, all_labels)

        # 返回多模态数据集
        dataset = {
            'stokes': all_stokes,
            'fluorescence': all_fluorescence,
            'images': all_images,
            'labels': all_labels,
            'class_map': self.class_map,
            'inverse_class_map': self.inverse_class_map,
            'sample_counts': sample_counts,
            'wavelengths': self.wavelengths,
            'signal_length': self.signal_length,
            'image_size': self.image_size,
            'num_views': self.num_views
        }

        return dataset

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

    def load_params(params_file='params.yaml'):
        """加载参数文件"""
        try:
            with open(params_file, 'r') as f:
                params = yaml.safe_load(f)
            return params
        except Exception as e:
            print(f"加载参数文件失败: {e}")
            return None

def main():
    # 加载参数
    params = MultimodalPolarFluDataPreprocessor.load_params()
    if params:
        # 从参数文件获取配置
        raw_data_path = Path(params['paths']['raw_data'])
        image_data_path = Path(params['paths']['image_data'])
        output_path = Path(params['paths']['output'])
        
        # 可以使用参数覆盖默认值
        signal_length = params['preprocess']['signal_length']
        rows_per_sample = params['preprocess']['rows_per_sample']
    else:
        # 使用默认路径
        raw_data_path = Path('/data3/zs/Multimodel_fusion/raw_data/polar_flu_data')
        image_data_path = Path('/data3/zs/AplimC/data/raw/images')
        output_path = Path('/data3/zs/Multimodel_fusion/processed_data')

    # 创建多模态预处理器实例
    preprocessor = MultimodalPolarFluDataPreprocessor(
        raw_data_path=raw_data_path,
        image_data_path=image_data_path,
        output_path=output_path
    )

    # 如果有参数，更新配置
    if params:
        preprocessor.signal_length = params['preprocess']['signal_length']
        preprocessor.rows_per_sample = params['preprocess']['rows_per_sample']
        preprocessor.image_size = tuple(params['preprocess']['image_size'])
        preprocessor.num_views = params['preprocess']['num_views']
        preprocessor.wavelengths = params['preprocess']['wavelengths']

    # 处理所有类别数据
    print("开始多模态数据预处理...")
    dataset = preprocessor.preprocess_all_classes()

    if dataset is not None:
        # 获取统计信息
        stats = preprocessor.get_label_statistics(dataset['labels'])
        print(f"\n多模态数据集统计信息:")
        for class_name, info in stats.items():
            print(f"{class_name}: {info['count']} 样本 ({info['percentage']:.1f}%)")

        # 保存数据集
        preprocessor.save_dataset(dataset)

        print(f"\n多模态数据预处理完成!")
        print(f"Stokes数据形状: {dataset['stokes'].shape}")
        print(f"荧光数据形状: {dataset['fluorescence'].shape}")
        print(f"图像数据形状: {dataset['images'].shape}")
        print(f"标签形状: {dataset['labels'].shape}")
        
        # 数据一致性最终检查
        print(f"\n最终一致性检查:")
        all_lengths = [len(dataset['stokes']), len(dataset['fluorescence']), len(dataset['images']), len(dataset['labels'])]
        print(f"所有模态样本数是否一致: {len(set(all_lengths)) == 1}")
    else:
        print("多模态数据预处理失败!")

if __name__ == "__main__":
    main()