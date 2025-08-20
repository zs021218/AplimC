#!/usr/bin/env python3
"""
信号统计量处理器
从原始HDF5文件中提取Stokes和Fluorescence信号，计算统计量，并保存为新的HDF5文件
"""

import numpy as np
import h5py
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import gc
import psutil
import os
from scipy import stats


class SignalStatisticsProcessor:
    """信号统计量处理器"""
    
    def __init__(self, input_hdf5: str, output_dir: str = None):
        self.input_hdf5 = Path(input_hdf5)
        if output_dir is None:
            output_dir = self.input_hdf5.parent
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 输出文件路径
        self.output_hdf5 = self.output_dir / 'multimodal_data_statistics.h5'
        
        # 统计量配置
        self.statistics_config = {
            'basic': ['mean', 'std', 'min', 'max', 'median'],
            'percentiles': [10, 25, 75, 90],
            'advanced': ['skewness', 'kurtosis', 'energy', 'zero_crossing_rate'],
            'frequency': ['peak_frequency', 'spectral_centroid', 'spectral_bandwidth']
        }
        
        print(f"初始化统计量处理器:")
        print(f"  输入文件: {self.input_hdf5}")
        print(f"  输出目录: {self.output_dir}")
        print(f"  输出文件: {self.output_hdf5}")
        
    def _check_memory(self) -> None:
        """检查内存使用并执行垃圾回收"""
        try:
            process = psutil.Process(os.getpid())
            memory_percent = process.memory_percent()
            if memory_percent > 80:
                print(f"内存使用率: {memory_percent:.1f}%, 执行垃圾回收")
                gc.collect()
        except:
            gc.collect()
    
    def _calculate_basic_statistics(self, signal: np.ndarray) -> np.ndarray:
        """计算基础统计量"""
        # signal shape: (n_samples, n_channels, signal_length)
        stats_list = []
        
        # 基础统计量
        stats_list.append(np.mean(signal, axis=2))      # 均值
        stats_list.append(np.std(signal, axis=2))       # 标准差
        stats_list.append(np.min(signal, axis=2))       # 最小值
        stats_list.append(np.max(signal, axis=2))       # 最大值
        stats_list.append(np.median(signal, axis=2))    # 中位数
        
        # 百分位数
        for p in self.statistics_config['percentiles']:
            stats_list.append(np.percentile(signal, p, axis=2))
        
        # 堆叠所有统计量 shape: (n_samples, n_channels, n_stats)
        return np.stack(stats_list, axis=2)
    
    def _calculate_advanced_statistics(self, signal: np.ndarray) -> np.ndarray:
        """计算高级统计量"""
        n_samples, n_channels, signal_length = signal.shape
        advanced_stats = []
        
        for i in range(n_samples):
            sample_stats = []
            for j in range(n_channels):
                channel_signal = signal[i, j, :]
                
                # 偏度和峰度
                skewness = stats.skew(channel_signal)
                kurtosis = stats.kurtosis(channel_signal)
                
                # 能量
                energy = np.sum(channel_signal ** 2)
                
                # 过零率
                zero_crossings = np.sum(np.diff(np.signbit(channel_signal)).astype(int))
                zero_crossing_rate = zero_crossings / (signal_length - 1)
                
                sample_stats.append([skewness, kurtosis, energy, zero_crossing_rate])
            
            advanced_stats.append(sample_stats)
            
            if i % 1000 == 0:
                self._check_memory()
        
        return np.array(advanced_stats, dtype=np.float32)
    
    def _calculate_frequency_statistics(self, signal: np.ndarray) -> np.ndarray:
        """计算频域统计量"""
        n_samples, n_channels, signal_length = signal.shape
        freq_stats = []
        
        # 计算频率轴
        freqs = np.fft.fftfreq(signal_length, d=1.0)
        freq_positive = freqs[:signal_length//2]
        
        for i in range(n_samples):
            sample_stats = []
            for j in range(n_channels):
                channel_signal = signal[i, j, :]
                
                # FFT
                fft_signal = np.fft.fft(channel_signal)
                power_spectrum = np.abs(fft_signal[:signal_length//2]) ** 2
                
                # 峰值频率
                peak_freq_idx = np.argmax(power_spectrum)
                peak_frequency = freq_positive[peak_freq_idx]
                
                # 谱质心
                spectral_centroid = np.sum(freq_positive * power_spectrum) / np.sum(power_spectrum)
                
                # 谱带宽
                spectral_bandwidth = np.sqrt(np.sum(((freq_positive - spectral_centroid) ** 2) * power_spectrum) / np.sum(power_spectrum))
                
                sample_stats.append([peak_frequency, spectral_centroid, spectral_bandwidth])
            
            freq_stats.append(sample_stats)
            
            if i % 1000 == 0:
                self._check_memory()
        
        return np.array(freq_stats, dtype=np.float32)
    
    def _process_signal_statistics(self, signal: np.ndarray, signal_name: str) -> np.ndarray:
        """处理单个信号的所有统计量"""
        print(f"  处理 {signal_name} 信号统计量...")
        print(f"    输入shape: {signal.shape}")
        
        # 计算各类统计量
        basic_stats = self._calculate_basic_statistics(signal)
        print(f"    基础统计量 shape: {basic_stats.shape}")
        
        advanced_stats = self._calculate_advanced_statistics(signal)
        print(f"    高级统计量 shape: {advanced_stats.shape}")
        
        freq_stats = self._calculate_frequency_statistics(signal)
        print(f"    频域统计量 shape: {freq_stats.shape}")
        
        # 合并所有统计量
        all_stats = np.concatenate([basic_stats, advanced_stats, freq_stats], axis=2)
        print(f"    合并后 shape: {all_stats.shape}")
        
        return all_stats.astype(np.float32)
    
    def _get_statistics_metadata(self) -> Dict:
        """获取统计量元数据"""
        feature_names = []
        
        # 基础统计量名称
        feature_names.extend(self.statistics_config['basic'])
        
        # 百分位数名称
        for p in self.statistics_config['percentiles']:
            feature_names.append(f'percentile_{p}')
        
        # 高级统计量名称
        feature_names.extend(self.statistics_config['advanced'])
        
        # 频域统计量名称
        feature_names.extend(self.statistics_config['frequency'])
        
        return {
            'feature_names': feature_names,
            'n_features': len(feature_names),
            'statistics_config': self.statistics_config
        }
    
    def _create_output_hdf5(self, input_file: h5py.File) -> h5py.File:
        """创建输出HDF5文件结构"""
        if self.output_hdf5.exists():
            self.output_hdf5.unlink()
        
        output_file = h5py.File(self.output_hdf5, 'w')
        
        # 获取原文件的基本信息
        sample_counts = {}
        for split_name in ['train', 'val', 'test']:
            if split_name in input_file:
                sample_counts[split_name] = len(input_file[split_name]['labels'])
        
        # 获取统计量元数据
        stats_metadata = self._get_statistics_metadata()
        n_features = stats_metadata['n_features']
        
        # 获取信号通道数
        stokes_channels = input_file['train']['stokes'].shape[1]  # 4
        fluorescence_channels = input_file['train']['fluorescence'].shape[1]  # 16
        
        print(f"统计量特征数: {n_features}")
        print(f"Stokes通道数: {stokes_channels}")
        print(f"Fluorescence通道数: {fluorescence_channels}")
        
        # 为每个分割创建数据集
        for split_name, n_samples in sample_counts.items():
            print(f"创建 {split_name} 组数据集 ({n_samples} 样本)...")
            
            split_group = output_file.create_group(split_name)
            
            # 创建统计量数据集
            chunk_size = min(100, n_samples)
            
            split_group.create_dataset(
                'stokes_stats', 
                (n_samples, stokes_channels, n_features),
                dtype=np.float32,
                chunks=(chunk_size, stokes_channels, n_features),
                compression='gzip',
                compression_opts=6
            )
            
            split_group.create_dataset(
                'fluorescence_stats',
                (n_samples, fluorescence_channels, n_features), 
                dtype=np.float32,
                chunks=(chunk_size, fluorescence_channels, n_features),
                compression='gzip',
                compression_opts=6
            )
            
            # 复制原始图像和标签数据
            if 'images' in input_file[split_name]:
                original_images = input_file[split_name]['images']
                split_group.create_dataset(
                    'images',
                    data=original_images[:],
                    chunks=True,
                    compression='gzip',
                    compression_opts=6
                )
            
            original_labels = input_file[split_name]['labels']
            split_group.create_dataset(
                'labels',
                data=original_labels[:],
                chunks=(chunk_size,),
                compression='gzip',
                compression_opts=6
            )
        
        # 保存元数据
        metadata_group = output_file.create_group('metadata')
        
        # 保存统计量元数据
        for key, value in stats_metadata.items():
            if isinstance(value, list):
                metadata_group.create_dataset(key, data=json.dumps(value).encode('utf-8'))
            elif isinstance(value, dict):
                metadata_group.create_dataset(key, data=json.dumps(value).encode('utf-8'))
            else:
                metadata_group.create_dataset(key, data=value)
        
        return output_file
    
    def process(self) -> bool:
        """主处理流程"""
        print("开始信号统计量处理...")
        
        try:
            # 打开输入文件
            with h5py.File(self.input_hdf5, 'r') as input_file:
                print(f"输入文件组: {list(input_file.keys())}")
                
                # 创建输出文件
                with self._create_output_hdf5(input_file) as output_file:
                    
                    # 处理每个分割
                    for split_name in ['train', 'val', 'test']:
                        if split_name not in input_file:
                            print(f"跳过不存在的分割: {split_name}")
                            continue
                        
                        print(f"\n处理 {split_name} 集...")
                        
                        # 读取原始信号数据
                        stokes_data = input_file[split_name]['stokes'][:]
                        fluorescence_data = input_file[split_name]['fluorescence'][:]
                        
                        print(f"  Stokes shape: {stokes_data.shape}")
                        print(f"  Fluorescence shape: {fluorescence_data.shape}")
                        
                        # 计算统计量
                        stokes_stats = self._process_signal_statistics(stokes_data, 'Stokes')
                        fluorescence_stats = self._process_signal_statistics(fluorescence_data, 'Fluorescence')
                        
                        # 写入输出文件
                        print(f"  写入统计量数据...")
                        output_file[split_name]['stokes_stats'][:] = stokes_stats
                        output_file[split_name]['fluorescence_stats'][:] = fluorescence_stats
                        
                        print(f"  {split_name} 集处理完成!")
                        
                        # 释放内存
                        del stokes_data, fluorescence_data, stokes_stats, fluorescence_stats
                        self._check_memory()
            
            # 生成统计量描述文件
            self._generate_statistics_description()
            
            print(f"\n统计量处理完成!")
            print(f"输出文件: {self.output_hdf5}")
            print(f"文件大小: {self.output_hdf5.stat().st_size / (1024**2):.2f} MB")
            
            return True
            
        except Exception as e:
            print(f"处理过程出错: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _generate_statistics_description(self) -> None:
        """生成统计量描述文件"""
        stats_metadata = self._get_statistics_metadata()
        
        description = {
            'file_info': {
                'input_file': str(self.input_hdf5),
                'output_file': str(self.output_hdf5),
                'creation_date': str(np.datetime64('now'))
            },
            'statistics_info': stats_metadata,
            'data_structure': {
                'groups': ['train', 'val', 'test', 'metadata'],
                'datasets_per_group': {
                    'train/val/test': [
                        'stokes_stats (n_samples, 4, n_features)',
                        'fluorescence_stats (n_samples, 16, n_features)', 
                        'images (n_samples, 3, 224, 224, 3)',
                        'labels (n_samples,)'
                    ],
                    'metadata': [
                        'feature_names',
                        'n_features', 
                        'statistics_config'
                    ]
                }
            },
            'feature_descriptions': {
                'basic_statistics': [
                    'mean: 信号均值',
                    'std: 信号标准差',
                    'min: 信号最小值',
                    'max: 信号最大值',
                    'median: 信号中位数'
                ],
                'percentiles': [f'percentile_{p}: {p}%分位数' for p in self.statistics_config['percentiles']],
                'advanced_statistics': [
                    'skewness: 偏度，衡量分布的偏斜程度',
                    'kurtosis: 峰度，衡量分布的尖锐程度',
                    'energy: 能量，信号平方和',
                    'zero_crossing_rate: 过零率，信号符号变化频率'
                ],
                'frequency_statistics': [
                    'peak_frequency: 峰值频率，功率谱最大值对应频率',
                    'spectral_centroid: 谱质心，频谱的重心',
                    'spectral_bandwidth: 谱带宽，频谱的分散程度'
                ]
            }
        }
        
        # 保存为JSON文件
        description_file = self.output_dir / 'statistics_description.json'
        with open(description_file, 'w', encoding='utf-8') as f:
            json.dump(description, f, indent=2, ensure_ascii=False)
        
        print(f"统计量描述已保存: {description_file}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='信号统计量处理器')
    parser.add_argument('--input', '-i', required=True, help='输入HDF5文件路径')
    parser.add_argument('--output', '-o', help='输出目录路径（默认为输入文件所在目录）')
    
    args = parser.parse_args()
    
    try:
        processor = SignalStatisticsProcessor(args.input, args.output)
        success = processor.process()
        
        if success:
            print("\n✅ 信号统计量处理完成!")
        else:
            print("\n❌ 处理失败!")
            return 1
            
    except Exception as e:
        print(f"❌ 处理过程出错: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    # 如果没有命令行参数，使用默认路径
    import sys
    if len(sys.argv) == 1:
        # 默认使用当前项目的HDF5文件
        input_file = "/data3/zs/AplimC/data/processed/multimodal_data.h5"
        if Path(input_file).exists():
            processor = SignalStatisticsProcessor(input_file)
            success = processor.process()
            exit(0 if success else 1)
        else:
            print(f"默认输入文件不存在: {input_file}")
            print("请使用: python signal_statistics_processor.py --input <HDF5文件路径>")
            exit(1)
    else:
        exit(main())
