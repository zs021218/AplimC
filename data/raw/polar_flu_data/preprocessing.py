from pathlib import Path
import numpy as np
import h5py

class PolarFluDataPreprocessor:
    def __init__(self, raw_data_path, output_path):
        self.raw_data_path = Path(raw_data_path)
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
        self.signal_length = 4000  # 修复拼写错误
        self.rows_per_sample = 20

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
                
                # 返回第一个数据集（假设只有一个主要数据集）
                if data_keys:
                    return mat_data[data_keys[0]]  # 修复：返回实际数据
                else:
                    print("未找到有效数据键")
                    return None
                    
        except Exception as e:
            print(f"加载文件失败 {mat_path}: {e}")
            return None
            
    def extract_samples(self, data):
        """从数据中提取样本"""
        if data is None:
            return None, None
            
        rows, cols = data.shape
        
        # 验证数据维度
        if cols != self.signal_length:
            print(f"警告: 信号长度不匹配。期望: {self.signal_length}, 实际: {cols}")

        # 样本数
        num_samples = rows // self.rows_per_sample
        print(f"提取样本数: {num_samples}")
        
        if num_samples == 0:
            print("警告: 数据行数不足以提取样本")
            return None, None
        
        # 初始化数组 
        stokes_data = np.zeros((num_samples, 4, cols))
        fluorescence_data = np.zeros((num_samples, 16, cols))  # 修正：16个通道

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
        
    def process_class_data(self, class_name):
        """处理特定类别的数据"""
        print(f"\n{'='*50}")
        print(f"处理 {class_name} 类别数据")
        print(f"{'='*50}")

        mat_file = self.raw_data_path / f"{class_name}.mat"
        if not mat_file.exists():
            print(f"未找到 {class_name} 类别的 .mat 文件: {mat_file}")
            return None

        print(f"处理文件: {mat_file.name}")

        data = self.load_mat_file(mat_file)
        if data is None:
            print(f"无法加载文件: {mat_file.name}")
            return None

        stokes_data, fluorescence_data = self.extract_samples(data)
        
        if stokes_data is None or fluorescence_data is None:
            print(f"无法从 {mat_file.name} 提取样本")
            return None

        # 创建标签
        class_label = self.get_class_label(class_name)
        if class_label == -1:
            print(f"警告: 未知类别 {class_name}")
            return None
            
        num_samples = stokes_data.shape[0]
        labels = np.full(num_samples, class_label, dtype=np.int32)

        print(f"类别标签: {class_label}")
        print(f"生成标签数量: {num_samples}")
        print(f"Stokes数据形状: {stokes_data.shape}")
        print(f"荧光数据形状: {fluorescence_data.shape}")

        return stokes_data, fluorescence_data, labels
        
    def preprocess_all_classes(self):
        """预处理所有类别的数据"""
        all_stokes = []
        all_fluorescence = []
        all_labels = []
        sample_counts = {}

        for class_name in self.class_map.keys():
            result = self.process_class_data(class_name)
            if result is None:
                sample_counts[class_name] = 0
                continue
                
            stokes_data, fluorescence_data, labels = result
            
            all_stokes.append(stokes_data)
            all_fluorescence.append(fluorescence_data)
            all_labels.append(labels)
            sample_counts[class_name] = len(labels)

        if not all_stokes:
            print("错误: 没有处理任何数据")
            return None

        # 合并所有类别的数据
        try:
            all_stokes = np.vstack(all_stokes)
            all_fluorescence = np.vstack(all_fluorescence)
            all_labels = np.concatenate(all_labels)
        except Exception as e:
            print(f"数据合并失败: {e}")
            return None

        print(f"\n{'='*50}")
        print(f"数据合并完成")
        print(f"{'='*50}")
        print(f"总样本数: {all_stokes.shape[0]}")
        print(f"Stokes数据形状: {all_stokes.shape}")
        print(f"荧光数据形状: {all_fluorescence.shape}")
        print(f"标签形状: {all_labels.shape}")

        # 返回数据集字典
        dataset = {
            'stokes': all_stokes,
            'fluorescence': all_fluorescence,
            'labels': all_labels,
            'class_map': self.class_map,
            'inverse_class_map': self.inverse_class_map,
            'sample_counts': sample_counts,
            'wavelengths': self.wavelengths,
            'signal_length': self.signal_length  # 修复：正确的属性名
        }

        return dataset
        
    def validate_labels(self, labels):
        """验证标签的有效性"""
        print(f"\n{'='*30}")
        print("标签验证")
        print(f"{'='*30}")
        
        # 检查标签范围
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
        
        print(f"标签验证完成")
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
        
    def save_dataset(self, dataset, filename='processed_data.npz'):
        """保存处理后的数据集"""
        save_path = self.output_path / filename
        
        try:
            np.savez_compressed(save_path, **dataset)
            print(f"数据集已保存到: {save_path}")
            
            # 保存元数据为JSON
            import json
            metadata = {
                'class_map': dataset['class_map'],
                'sample_counts': dataset['sample_counts'],
                'wavelengths': dataset['wavelengths'],
                'signal_length': dataset['signal_length'],
                'total_samples': int(len(dataset['labels']))
            }
            
            metadata_path = self.output_path / 'metadata.json'
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            print(f"元数据已保存到: {metadata_path}")
            
        except Exception as e:
            print(f"保存失败: {e}")

    def load_dataset(self, filename='processed_data.npz'):
        """加载处理后的数据集"""
        load_path = self.output_path / filename
        if not load_path.exists():
            print(f"文件不存在: {load_path}")
            return None
        
        try:
            data = np.load(load_path, allow_pickle=True)
            dataset = {key: data[key] for key in data.files}
            print(f"数据集已从 {load_path} 加载")
            return dataset
        except Exception as e:
            print(f"加载失败: {e}")
            return None

def main():
    raw_data_path = Path('/data3/zs/Multimodel_fusion/raw_data/polar_flu_data')
    output_path = Path('/data3/zs/Multimodel_fusion/processed_data')

    # 创建预处理器实例
    preprocessor = PolarFluDataPreprocessor(raw_data_path, output_path)

    # 处理所有类别数据
    print("开始数据预处理...")
    dataset = preprocessor.preprocess_all_classes()

    if dataset is not None:
       # 验证标签
       preprocessor.validate_labels(dataset['labels'])

       # 获取统计信息
       stats = preprocessor.get_label_statistics(dataset['labels'])
       print(f"\n数据集统计信息:")
       for class_name, info in stats.items():
           print(f"{class_name}: {info['count']} 样本 ({info['percentage']:.1f}%)")

       # 保存数据集
       preprocessor.save_dataset(dataset)

       print(f"\n数据预处理完成!")
       print(f"Stokes数据形状: {dataset['stokes'].shape}")
       print(f"荧光数据形状: {dataset['fluorescence'].shape}")
       print(f"标签形状: {dataset['labels'].shape}")
    else:
       print("数据预处理失败!")

if __name__ == "__main__":
    main()




