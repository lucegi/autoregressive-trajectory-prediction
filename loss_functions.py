"""
Author: Giovanni Lucente, Marko Mizdrak

This script contains possible loss functions for the training algorithm.

"""
import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
import torch.nn.functional as F
import torchvision.models as models
import pywt 

def balanced_weighted_l1_loss(output, target, threshold=0.05, high_weight=50.0, low_weight=1.0, false_positive_weight=50.0):
    """
    Computes a balanced weighted L1 loss to penalize both false negatives (missed color)
    and false positives (wrongly added color).
    """
    # Assign higher weight to target-colored pixels
    weight = torch.where(target > threshold, high_weight, low_weight)

    # Assign high penalty for pixels that the network colored but should be black
    false_positive_mask = (target <= threshold) & (output > threshold)
    weight[false_positive_mask] = false_positive_weight

    loss = (weight * torch.abs(output - target)).mean()
    return loss
    
def weighted_l1_loss(output, target, threshold = 0.05, high_weight=50.0, low_weight=1.0):
    """
    Computes a weighted L1 loss to address data imbalance.
    """
    weight = torch.where(target > threshold, high_weight, low_weight)
    loss = (weight * torch.abs(output - target)).mean()
    return loss

def weighted_l2_loss(output, target, threshold=0.05, high_weight=50.0, low_weight=1.0):
    """
    Computes a weighted L2 loss to address data imbalance.
    """
    weight = torch.where(target > threshold, high_weight, low_weight)
    loss = (weight * (output - target) ** 2).mean()  # L2 loss is squared error
    return loss

def ssim(pred, target, window_size=11, reduction='mean'):
    def create_window(window_size, channel):
        sigma = 1.5
        kernel = torch.arange(window_size).float() - (window_size - 1) / 2.0
        kernel = torch.exp(-0.5 * (kernel / sigma) ** 2)
        kernel /= kernel.sum()
        kernel_2d = kernel[:, None] * kernel[None, :]
        window = kernel_2d.expand(channel, 1, window_size, window_size)
        return window / window.sum()

    _, c, h, w = pred.size()
    assert min(h, w) >= window_size, f"Input image must be larger than window size {window_size}"

    window = create_window(window_size, c).to(pred.device)

    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=c)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=c)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=c) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=c) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=c) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if reduction == 'mean':
        return ssim_map.mean()
    elif reduction == 'sum':
        return ssim_map.sum()
    else:
        return ssim_map

def ssim_loss(pred, target, window_size=11):
    return 1 - ssim(pred, target, window_size)

def dice_loss(pred, target, smooth=1e-6):
    # Apply a Sigmoid to the predictions (optional, depending on your model's output)
    pred = torch.sigmoid(pred)  # Ensure predictions are in [0, 1]
    
    # Calculate intersection and union
    intersection = (pred * target).sum()  # Element-wise multiplication
    union = pred.sum() + target.sum()
    
    # Compute the Dice coefficient
    dice = (2 * intersection + smooth) / (union + smooth)
    
    # Return Dice loss (1 - Dice coefficient)
    return 1 - dice

def weighted_dice_loss(pred, target, smooth=1e-6, threshold = 0.05, high_weight=50.0, low_weight=1.0):

    weight = torch.where(target > threshold, high_weight, low_weight)

    # Apply sigmoid to scale predictions between [0, 1]
    pred = torch.sigmoid(pred)
    
    # Weighted intersection and union
    intersection = (weight * pred * target).sum()  # Element-wise multiplication with weight
    union = (weight * pred).sum() + (weight * target).sum()
    
    # Compute weighted Dice coefficient
    dice = (2 * intersection + smooth) / (union + smooth)
    
    # Return Dice loss
    return 1 - dice

def weighted_ssim(pred, target, window_size=11, reduction='mean', threshold = 0.05, high_weight=50.0, low_weight=1.0):
    def create_window(window_size, channel):
        sigma = 1.5
        kernel = torch.arange(window_size).float() - (window_size - 1) / 2.0
        kernel = torch.exp(-0.5 * (kernel / sigma) ** 2)
        kernel /= kernel.sum()
        kernel_2d = kernel[:, None] * kernel[None, :]
        window = kernel_2d.expand(channel, 1, window_size, window_size)
        return window / window.sum()

    _, c, h, w = pred.size()
    assert min(h, w) >= window_size, f"Input image must be larger than window size {window_size}"

    window = create_window(window_size, c).to(pred.device)

    # SSIM computation
    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=c)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=c)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=c) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=c) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=c) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    weights = torch.where(target > threshold, high_weight, low_weight)

    # Apply weights to the SSIM map
    weighted_ssim_map = ssim_map * weights  # Element-wise weighting

    # Reduction
    if reduction == 'mean':
        return weighted_ssim_map.sum() / weights.sum()
    elif reduction == 'sum':
        return weighted_ssim_map.sum()
    else:
        return weighted_ssim_map

def weighted_ssim_loss(pred, target):
    return 1 - weighted_ssim(pred, target)

class PerceptualLoss(nn.Module):
    def __init__(self, device="cuda"):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features[:16].eval()  # Use the first layers for feature extraction
        for param in vgg.parameters():
            param.requires_grad = False  # Freeze VGG weights
        
        self.vgg = vgg.to(device)  # Move model to the correct device
        self.criterion = nn.L1Loss()
        self.device = device

    def forward(self, pred, target):
        pred = pred.to(self.device)  # Ensure inputs are on the same device as VGG
        target = target.to(self.device)

        pred_features = self.vgg(pred)
        target_features = self.vgg(target)

        return self.criterion(pred_features, target_features)

def sobel_edges(image):
    # Define Sobel kernels for edge detection
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=image.dtype, device=image.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=image.dtype, device=image.device).view(1, 1, 3, 3)

    # Expand kernels to match the number of input channels
    c = image.shape[1]  # Number of channels
    sobel_x = sobel_x.repeat(c, 1, 1, 1)  # Repeat kernel for each channel
    sobel_y = sobel_y.repeat(c, 1, 1, 1)

    # Apply Sobel kernels
    edge_x = F.conv2d(image, sobel_x, padding=1, groups=c)
    edge_y = F.conv2d(image, sobel_y, padding=1, groups=c)

    # Combine edge responses
    edges = torch.sqrt(edge_x**2 + edge_y**2)
    return edges

def edge_loss(pred, target):
    pred_edges = sobel_edges(pred)
    target_edges = sobel_edges(target)
    return F.l1_loss(pred_edges, target_edges)

def color_loss(pred, target):
    return F.mse_loss(pred, target)

class WaveletLoss(nn.Module):
    def __init__(self, base_loss=nn.L1Loss(), wavelet='haar', device="cuda"):
        super(WaveletLoss, self).__init__()
        self.base_loss = base_loss
        self.wavelet = wavelet
        self.device = device  # Store device information

    def forward(self, pred, target):
        # Ensure inputs are on the correct device
        pred = pred.to(self.device)
        target = target.to(self.device)

        # Compute loss on original images
        loss = self.base_loss(pred, target)

        # Apply wavelet decomposition
        for _ in range(3):  # Multi-scale loss (3 levels of decomposition)
            # Convert tensors to numpy (must be moved to CPU first)
            pred_np = pred.cpu().detach().numpy()
            target_np = target.cpu().detach().numpy()

            # Perform wavelet decomposition
            pred_np, (cH1, cV1, cD1) = pywt.dwt2(pred_np, self.wavelet)
            target_np, (cH2, cV2, cD2) = pywt.dwt2(target_np, self.wavelet)

            # Convert back to PyTorch tensors and move to the original device
            pred = torch.tensor(pred_np, dtype=pred.dtype, device=self.device)
            cH1 = torch.tensor(cH1, dtype=pred.dtype, device=self.device)
            cV1 = torch.tensor(cV1, dtype=pred.dtype, device=self.device)
            cD1 = torch.tensor(cD1, dtype=pred.dtype, device=self.device)

            target = torch.tensor(target_np, dtype=target.dtype, device=self.device)
            cH2 = torch.tensor(cH2, dtype=target.dtype, device=self.device)
            cV2 = torch.tensor(cV2, dtype=target.dtype, device=self.device)
            cD2 = torch.tensor(cD2, dtype=target.dtype, device=self.device)

            # Compute loss at each level
            loss += (self.base_loss(cH1, cH2) + self.base_loss(cV1, cV2) + self.base_loss(cD1, cD2))

        return loss

import torch
import torch.nn.functional as F

def VAE_loss_function(recon_x, x, mu, logvar, beta=1.0):
    """
    Loss function for Conditional VAE with weighted reconstruction loss.
    
    Parameters:
    - recon_x: Reconstructed image from decoder
    - x: Original input image
    - mu: Mean of the latent distribution (output from encoder)
    - logvar: Log variance of the latent distribution (output from encoder)
    - beta: Weight for KL divergence term (can be used for β-VAE)
    - weight_factor: Scaling factor for the pixel-wise weights
    
    Returns:
    - Total loss: The sum of the weighted reconstruction loss and the KL divergence loss
    """

    # Compute element-wise reconstruction loss
    #recon_loss = weighted_l1_loss(recon_x, x)
    recon_loss = color_loss(recon_x, x)

    #print("Mean of x:", torch.mean(x).item())
    #print("Mean of recon_x:", torch.mean(recon_x).item())
    #print("Max of recon_x:", torch.max(recon_x).item())
    #print("Min of recon_x:", torch.min(recon_x).item())

    # KL Divergence Loss (Gaussian)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Compute total loss
    total_loss = recon_loss + beta * kl_loss

    return total_loss
