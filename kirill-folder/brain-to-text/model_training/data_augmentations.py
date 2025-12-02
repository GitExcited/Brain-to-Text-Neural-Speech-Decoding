import torch
import torch.nn.functional as F
import torchaudio.transforms as T
import numpy as np
from scipy.ndimage import gaussian_filter1d

def gauss_smooth(inputs, device, smooth_kernel_std=2, smooth_kernel_size=100,  padding='same'):
    """
    Applies a 1D Gaussian smoothing operation with PyTorch to smooth the data along the time axis.
    Args:
        inputs (tensor : B x T x N): A 3D tensor with batch size B, time steps T, and number of features N.
                                     Assumed to already be on the correct device (e.g., GPU).
        kernelSD (float): Standard deviation of the Gaussian smoothing kernel.
        padding (str): Padding mode, either 'same' or 'valid'.
        device (str): Device to use for computation (e.g., 'cuda' or 'cpu').
    Returns:
        smoothed (tensor : B x T x N): A smoothed 3D tensor with batch size B, time steps T, and number of features N.
    """
    # Get Gaussian kernel
    inp = np.zeros(smooth_kernel_size, dtype=np.float32)
    inp[smooth_kernel_size // 2] = 1
    gaussKernel = gaussian_filter1d(inp, smooth_kernel_std)
    validIdx = np.argwhere(gaussKernel > 0.01)
    gaussKernel = gaussKernel[validIdx]
    gaussKernel = np.squeeze(gaussKernel / np.sum(gaussKernel))

    # Convert to tensor
    gaussKernel = torch.tensor(gaussKernel, dtype=torch.float32, device=device)
    gaussKernel = gaussKernel.view(1, 1, -1)  # [1, 1, kernel_size]

    # Prepare convolution
    B, T, C = inputs.shape
    inputs = inputs.permute(0, 2, 1)  # [B, C, T]
    gaussKernel = gaussKernel.repeat(C, 1, 1)  # [C, 1, kernel_size]

    # Perform convolution
    smoothed = F.conv1d(inputs, gaussKernel, padding=padding, groups=C)
    return smoothed.permute(0, 2, 1)  # [B, T, C]




def spec_augment_torchaudio(inputs, device, freq_mask_param=10, time_mask_param=15, 
                            n_freq_masks=2, n_time_masks=2):
    """
    SpecAugment using torchaudio's built-in transforms.
    
    Reference:
        Park, D. S., et al. "SpecAugment: A Simple Data Augmentation Method for 
        Automatic Speech Recognition." Interspeech 2019.
        
    Implementation:
        Uses torchaudio.transforms.FrequencyMasking and TimeMasking
        https://pytorch.org/audio/stable/transforms.html
    
    Args:
        inputs (tensor): Shape (B, T, N) - batch, time, features
        device: torch device
        freq_mask_param (int): Maximum frequency mask width
        time_mask_param (int): Maximum time mask width
        n_freq_masks (int): Number of frequency masks
        n_time_masks (int): Number of time masks
    
    Returns:
        augmented (tensor): Same shape as input
    """
    # torchaudio expects (batch, features, time) but we have (batch, time, features)
    # So we need to transpose
    x = inputs.transpose(1, 2)  # (B, T, N) -> (B, N, T)
    
    # Apply frequency masking
    freq_masker = T.FrequencyMasking(freq_mask_param=freq_mask_param)
    for _ in range(n_freq_masks):
        x = freq_masker(x)
    
    # Apply time masking
    time_masker = T.TimeMasking(time_mask_param=time_mask_param)
    for _ in range(n_time_masks):
        x = time_masker(x)
    
    # Transpose back to (B, T, N)
    return x.transpose(1, 2)