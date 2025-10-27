"""
GWNet: GammaWaveletNet for Low-Light Image Enhancement
Implementation based on the research paper

Paper: GWNet: A Lightweight Model for Low-Light Image Enhancement 
       Using Gamma Correction and Wavelet Transform
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np


class GammaCorrection(nn.Module):
    """
    Gamma Correction Module
    Applies adaptive gamma correction to enhance low-light images
    """
    def __init__(self):
        super(GammaCorrection, self).__init__()
        # Learnable gamma parameter
        self.gamma = nn.Parameter(torch.tensor(2.2))
        
    def forward(self, x):
        # Normalize to [0, 1]
        x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8)
        # Apply gamma correction
        corrected = torch.pow(x_norm, 1.0 / self.gamma)
        return corrected


class DWT2D(nn.Module):
    """
    2D Discrete Wavelet Transform using Haar wavelet
    Decomposes image into LL (low-freq) and LH, HL, HH (high-freq) components
    """
    def __init__(self, wavelet='haar'):
        super(DWT2D, self).__init__()
        self.wavelet = wavelet
        
    def forward(self, x):
        batch, channels, height, width = x.shape
        
        # Apply DWT to each channel
        LL_list, LH_list, HL_list, HH_list = [], [], [], []
        
        for i in range(batch):
            for c in range(channels):
                img = x[i, c].cpu().detach().numpy()
                coeffs = pywt.dwt2(img, self.wavelet)
                cA, (cH, cV, cD) = coeffs
                
                LL_list.append(torch.from_numpy(cA).unsqueeze(0))
                LH_list.append(torch.from_numpy(cH).unsqueeze(0))
                HL_list.append(torch.from_numpy(cV).unsqueeze(0))
                HH_list.append(torch.from_numpy(cD).unsqueeze(0))
        
        # Stack and reshape
        LL = torch.stack(LL_list).view(batch, channels, -1, cA.shape[1]).to(x.device)
        LH = torch.stack(LH_list).view(batch, channels, -1, cH.shape[1]).to(x.device)
        HL = torch.stack(HL_list).view(batch, channels, -1, cV.shape[1]).to(x.device)
        HH = torch.stack(HH_list).view(batch, channels, -1, cD.shape[1]).to(x.device)
        
        return LL, (LH, HL, HH)


class IDWT2D(nn.Module):
    """
    2D Inverse Discrete Wavelet Transform
    Reconstructs image from wavelet coefficients
    """
    def __init__(self, wavelet='haar'):
        super(IDWT2D, self).__init__()
        self.wavelet = wavelet
        
    def forward(self, LL, LH, HL, HH, target_shape):
        batch, channels, _, _ = LL.shape
        
        reconstructed = []
        for i in range(batch):
            for c in range(channels):
                cA = LL[i, c].cpu().detach().numpy()
                cH = LH[i, c].cpu().detach().numpy()
                cV = HL[i, c].cpu().detach().numpy()
                cD = HH[i, c].cpu().detach().numpy()
                
                img = pywt.idwt2((cA, (cH, cV, cD)), self.wavelet)
                reconstructed.append(torch.from_numpy(img).unsqueeze(0))
        
        result = torch.stack(reconstructed).view(batch, channels, target_shape[2], target_shape[3])
        return result.to(LL.device)


class SpatialWaveletInteraction(nn.Module):
    """
    Spatial Wavelet Interaction (SWI) Component
    Combines spatial and frequency domain information
    """
    def __init__(self, in_channels, out_channels):
        super(SpatialWaveletInteraction, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        return out


class UNetBlock(nn.Module):
    """
    U-Net style encoder-decoder block
    """
    def __init__(self, in_channels, out_channels):
        super(UNetBlock, self).__init__()
        
        # Encoder
        self.enc1 = SpatialWaveletInteraction(in_channels, 64)
        self.enc2 = SpatialWaveletInteraction(64, 128)
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = SpatialWaveletInteraction(128, 256)
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec1 = SpatialWaveletInteraction(256, 128)
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = SpatialWaveletInteraction(128, 64)
        
        # Output
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e2))
        
        # Decoder with skip connections
        d1 = self.up1(b)
        d1 = torch.cat([d1, e2], dim=1)
        d1 = self.dec1(d1)
        
        d2 = self.up2(d1)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)
        
        out = self.out_conv(d2)
        return out


class LSubnetwork(nn.Module):
    """
    L Subnetwork: Processes low-frequency components (LL)
    """
    def __init__(self, in_channels=3):
        super(LSubnetwork, self).__init__()
        self.unet = UNetBlock(in_channels, in_channels)
        
    def forward(self, LL):
        return self.unet(LL)


class HSubnetwork(nn.Module):
    """
    H Subnetwork: Refines high-frequency components (LH, HL, HH)
    """
    def __init__(self, in_channels=3):
        super(HSubnetwork, self).__init__()
        # Process each high-freq component
        self.unet_LH = UNetBlock(in_channels, in_channels)
        self.unet_HL = UNetBlock(in_channels, in_channels)
        self.unet_HH = UNetBlock(in_channels, in_channels)
        
    def forward(self, LH, HL, HH):
        LH_refined = self.unet_LH(LH)
        HL_refined = self.unet_HL(HL)
        HH_refined = self.unet_HH(HH)
        return LH_refined, HL_refined, HH_refined


class GWNet(nn.Module):
    """
    Complete GammaWaveletNet Model
    
    Architecture:
    1. Gamma Correction Module
    2. 2D Discrete Wavelet Transform (DWT)
    3. L Subnetwork (processes LL - low frequency)
    4. H Subnetwork (refines LH, HL, HH - high frequency)
    5. Inverse DWT (IDWT)
    """
    def __init__(self):
        super(GWNet, self).__init__()
        
        self.gamma_correction = GammaCorrection()
        self.dwt = DWT2D(wavelet='haar')
        self.idwt = IDWT2D(wavelet='haar')
        
        self.l_subnet = LSubnetwork(in_channels=3)
        self.h_subnet = HSubnetwork(in_channels=3)
        
    def forward(self, x):
        original_shape = x.shape
        
        # Step 1: Gamma Correction
        gamma_corrected = self.gamma_correction(x)
        
        # Step 2: Wavelet Decomposition
        LL, (LH, HL, HH) = self.dwt(gamma_corrected)
        
        # Step 3: Process low-frequency with L-Subnetwork
        LL_enhanced = self.l_subnet(LL)
        
        # Step 4: Refine high-frequency with H-Subnetwork
        LH_refined, HL_refined, HH_refined = self.h_subnet(LH, HL, HH)
        
        # Step 5: Inverse Wavelet Transform
        enhanced = self.idwt(LL_enhanced, LH_refined, HL_refined, HH_refined, original_shape)
        
        return enhanced


def count_parameters(model):
    """Count the number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    model = GWNet()
    print(f"GWNet Model Parameters: {count_parameters(model):,}")
    
    # Test with dummy input
    dummy_input = torch.randn(1, 3, 256, 256)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
