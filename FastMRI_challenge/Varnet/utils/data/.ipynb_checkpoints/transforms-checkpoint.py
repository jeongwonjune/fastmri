import numpy as np
import torch

def to_tensor(data):
    """
    Convert numpy array to PyTorch tensor. For complex arrays, the real and imaginary parts
    are stacked along the last dimension.
    Args:
        data (np.array): Input numpy array
    Returns:
        torch.Tensor: PyTorch version of data
    """
    return torch.from_numpy(data)

class DataTransform:
    def __init__(self, augmentor, isforward, max_key, use_seed: bool = True):
        self.isforward = isforward
        self.max_key = max_key
        self.use_seed = use_seed
        if augmentor is not None:
            self.use_augment = True
            self.augmentor = augmentor
        else:
            self.use_augment = False
            
    def __call__(self, mask, input, target, attrs, fname, slice):
        
        seed = None if not self.use_seed else tuple(map(ord, fname))
        
        if not self.isforward and self.use_augment:
            if self.augmentor.schedule_p() > 0.0: #train set, augmentation probability 가 0보다 클 때
                input = to_tensor(input)
                input = torch.stack((input.real, input.imag), dim=-1) # 복소수 텐서를 실수 2개로 이루어진 텐서로 변환
                input, target = self.augmentor(input, target.shape) #augmentation 진행
                mask = torch.from_numpy(mask.reshape(1, 1, input.shape[-2], 1).astype(np.float32)).byte()
                kspace = input * mask
                
            else:
                kspace = to_tensor(input * mask)
                kspace = torch.stack((kspace.real, kspace.imag), dim=-1)
                mask = torch.from_numpy(mask.reshape(1, 1, kspace.shape[-2], 1).astype(np.float32)).byte()
                target = to_tensor(target)
            maximum = attrs[self.max_key]
            
        elif not self.isforward and not self.use_augment: # validation set
            kspace = to_tensor(input * mask)
            kspace = torch.stack((kspace.real, kspace.imag), dim=-1)
            mask = torch.from_numpy(mask.reshape(1, 1, kspace.shape[-2], 1).astype(np.float32)).byte()
            target = to_tensor(target)
            maximum = attrs[self.max_key]
            
        else: # test set
            kspace = to_tensor(input * mask)
            kspace = torch.stack((kspace.real, kspace.imag), dim=-1)
            mask = torch.from_numpy(mask.reshape(1, 1, kspace.shape[-2], 1).astype(np.float32)).byte()
            target = -1
            maximum = -1
            
      
               
            
        return mask, kspace, target, maximum, fname, slice
    
    
    
    def seed_pipeline(self, seed):
        """
        Sets random seed for the MRAugment pipeline. It is important to provide
        different seed to different workers and across different GPUs to keep
        the augmentations diverse.
        
        For an example how to set it see worker_init in pl_modules/fastmri_data_module.py
        """
        if self.use_augment:
            if self.augmentor.aug_on:
                self.augmentor.augmentation_pipeline.rng.seed(seed)