"""
数据处理模块

该模块包含了多模态偏振荧光数据的数据集定义、数据加载器和数据变换。
"""

from .dataset import MultimodalPolarFluDataset
from .dataloader import MultimodalDataLoader, create_dataloaders
from .transforms import get_transforms, get_signal_transforms, SignalAugmentation

__all__ = [
    'MultimodalPolarFluDataset',
    'MultimodalDataLoader', 
    'create_dataloaders',
    'get_transforms',
    'get_signal_transforms',
    'SignalAugmentation'
]

from .dataset import (
    MultimodalPolarFluDataset,
    get_dataset_statistics,
    create_dataset
)

from .dataloader import (
    MultimodalDataLoader,
    create_dataloaders,
    get_class_weights
)

from .transforms import (
    get_image_transforms,
    get_signal_transforms,
    MultimodalTransform
)

__all__ = [
    # Dataset classes and utilities
    'MultimodalPolarFluDataset',
    'get_dataset_statistics', 
    'create_dataset',
    
    # DataLoader classes and utilities
    'MultimodalDataLoader',
    'create_dataloaders',
    'get_class_weights',
    
    # Transform functions
    'get_image_transforms',
    'get_signal_transforms',
    'MultimodalTransform',
]

# Package metadata
__version__ = '1.0.0'
__author__ = 'AplimC Team'
__description__ = 'Multimodal polar fluorescence data handling package'

def get_package_info():
    """Get package information"""
    return {
        'name': 'AplimC.data',
        'version': __version__,
        'author': __author__,
        'description': __description__,
        'components': [
            'dataset - PyTorch Dataset classes',
            'dataloader - DataLoader utilities', 
            'transforms - Data transformation functions'
        ]
    }

# Convenience function for quick dataset creation
def quick_setup(data_path, batch_size=32, num_workers=4, **kwargs):
    """
    Quick setup function for creating train/val/test dataloaders
    
    Args:
        data_path (str): Path to processed data
        batch_size (int): Batch size for dataloaders
        num_workers (int): Number of worker processes
        **kwargs: Additional arguments for dataloader creation
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    return create_dataloaders(
        data_path=data_path,
        batch_size=batch_size, 
        num_workers=num_workers,
        **kwargs
    )