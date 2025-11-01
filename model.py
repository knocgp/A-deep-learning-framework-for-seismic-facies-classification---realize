"""
Models for Seismic Facies Classification
Based on: "A deep learning framework for seismic facies classification" (Kaur et al., 2022)

Implements:
1. DeepLabv3+ with modified Xception backbone
2. GAN-based segmentation network (Generator + Discriminator)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ============================================================================
# DeepLabv3+ Implementation
# ============================================================================

class SeparableConv2d(nn.Module):
    """
    Depthwise Separable Convolution.
    Used in Xception and DeepLabv3+ for efficiency.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 padding=1, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        
        # Depthwise convolution
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride,
            padding, dilation, groups=in_channels, bias=bias
        )
        
        # Pointwise convolution
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias
        )
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class ASPPModule(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP) module.
    Captures multi-scale contextual information using parallel atrous convolutions.
    """
    def __init__(self, in_channels, out_channels, atrous_rates=[6, 12, 18]):
        super(ASPPModule, self).__init__()
        
        modules = []
        
        # 1x1 convolution
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        
        # Atrous convolutions with different rates
        for rate in atrous_rates:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, 
                         dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        
        # Global average pooling
        modules.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        
        self.convs = nn.ModuleList(modules)
        
        # Project concatenated features
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
    
    def forward(self, x):
        res = []
        for conv in self.convs:
            if isinstance(conv[0], nn.AdaptiveAvgPool2d):
                # Upsample global pooling features
                res.append(F.interpolate(
                    conv(x), size=x.shape[2:], mode='bilinear', align_corners=False
                ))
            else:
                res.append(conv(x))
        
        res = torch.cat(res, dim=1)
        return self.project(res)


class XceptionBlock(nn.Module):
    """
    Modified Xception block with separable convolutions.
    """
    def __init__(self, in_channels, out_channels, stride=1, dilation=1, 
                 use_skip=True, use_relu_first=True):
        super(XceptionBlock, self).__init__()
        
        self.use_skip = use_skip
        self.use_relu_first = use_relu_first
        
        if use_skip:
            if stride != 1 or in_channels != out_channels:
                self.skip = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.skip = None
        
        # Three separable convolutions
        self.conv1 = SeparableConv2d(in_channels, out_channels, 3, 1, dilation, dilation)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = SeparableConv2d(out_channels, out_channels, 3, 1, dilation, dilation)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = SeparableConv2d(out_channels, out_channels, 3, stride, dilation, dilation)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        
        if self.use_relu_first:
            x = self.relu(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        
        if self.use_skip:
            if self.skip is not None:
                residual = self.skip(residual)
            x = x + residual
        
        return x


class ModifiedXception(nn.Module):
    """
    Modified Xception encoder as described in DeepLabv3+.
    Replaces max pooling with atrous separable convolutions.
    """
    def __init__(self, in_channels=1, output_stride=16):
        super(ModifiedXception, self).__init__()
        
        # Entry flow
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.block1 = XceptionBlock(64, 128, stride=2)
        self.block2 = XceptionBlock(128, 256, stride=2)
        self.block3 = XceptionBlock(256, 728, stride=2)
        
        # Middle flow (repeat 8 times in original, using 4 for efficiency)
        # Comment: Reduced from 8 to 4 repetitions for computational efficiency
        # Original paper doesn't specify exact number for seismic application
        self.middle_flow = nn.Sequential(*[
            XceptionBlock(728, 728, stride=1) for _ in range(4)
        ])
        
        # Exit flow with atrous convolutions
        if output_stride == 16:
            self.block4 = XceptionBlock(728, 1024, stride=1, dilation=2)
        else:
            self.block4 = XceptionBlock(728, 1024, stride=2, dilation=1)
        
        self.block5 = XceptionBlock(1024, 1536, stride=1, dilation=2, use_skip=False)
        self.block6 = XceptionBlock(1536, 2048, stride=1, dilation=2, use_skip=False)
    
    def forward(self, x):
        # Entry flow
        x = self.conv1(x)
        x = self.conv2(x)
        low_level_feat = x  # Low-level features for decoder
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        # Middle flow
        x = self.middle_flow(x)
        
        # Exit flow
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        
        return x, low_level_feat


class DeepLabV3Plus(nn.Module):
    """
    DeepLabv3+ for seismic facies segmentation.
    
    Args:
        in_channels: Number of input channels (1 for grayscale seismic)
        num_classes: Number of facies classes (6 as per paper)
        output_stride: Output stride for encoder (8 or 16)
    """
    def __init__(self, in_channels=1, num_classes=6, output_stride=16):
        super(DeepLabV3Plus, self).__init__()
        
        self.num_classes = num_classes
        
        # Encoder (Modified Xception)
        self.encoder = ModifiedXception(in_channels, output_stride)
        
        # ASPP
        self.aspp = ASPPModule(2048, 256)
        
        # Decoder
        self.decoder_conv1 = nn.Sequential(
            nn.Conv2d(64, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        self.decoder_conv2 = nn.Sequential(
            SeparableConv2d(256 + 48, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            SeparableConv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Final classifier
        self.classifier = nn.Conv2d(256, num_classes, 1)
        
        # Dropout for uncertainty estimation (Bayesian approximation)
        self.dropout = nn.Dropout2d(p=0.5)
    
    def forward(self, x, use_dropout=False):
        input_size = x.shape[2:]
        
        # Encoder
        encoder_out, low_level_feat = self.encoder(x)
        
        # ASPP
        x = self.aspp(encoder_out)
        
        # Optional dropout for uncertainty estimation
        if use_dropout:
            x = self.dropout(x)
        
        # Upsample encoder features
        x = F.interpolate(x, size=low_level_feat.shape[2:], 
                         mode='bilinear', align_corners=False)
        
        # Process low-level features
        low_level_feat = self.decoder_conv1(low_level_feat)
        
        # Concatenate
        x = torch.cat([x, low_level_feat], dim=1)
        
        # Decoder convolutions
        x = self.decoder_conv2(x)
        
        if use_dropout:
            x = self.dropout(x)
        
        # Final classification
        x = self.classifier(x)
        
        # Upsample to input size
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        
        return x


# ============================================================================
# GAN-based Segmentation Implementation
# ============================================================================

class UNetGenerator(nn.Module):
    """
    U-Net style generator for GAN-based segmentation.
    Uses encoder-decoder architecture with skip connections.
    """
    def __init__(self, in_channels=1, num_classes=6):
        super(UNetGenerator, self).__init__()
        
        # Encoder
        self.enc1 = self._conv_block(in_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self._conv_block(512, 1024)
        
        # Decoder
        self.dec4 = self._upconv_block(1024, 512)
        self.dec4_conv = self._conv_block(1024, 512)  # 512 + 512 = 1024
        
        self.dec3 = self._upconv_block(512, 256)
        self.dec3_conv = self._conv_block(512, 256)  # 256 + 256 = 512
        
        self.dec2 = self._upconv_block(256, 128)
        self.dec2_conv = self._conv_block(256, 128)  # 128 + 128 = 256
        
        self.dec1 = self._upconv_block(128, 64)
        self.dec1_conv = self._conv_block(128, 64)   # 64 + 64 = 128
        
        # Final layer
        self.final = nn.Conv2d(64, num_classes, 1)
        
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout2d(p=0.5)
    
    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def _upconv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, use_dropout=False):
        # Encoder with skip connections
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))
        
        if use_dropout:
            bottleneck = self.dropout(bottleneck)
        
        # Decoder with skip connections
        dec4 = self.dec4(bottleneck)
        # Crop enc4 to match dec4 size if needed
        if enc4.size()[2:] != dec4.size()[2:]:
            enc4 = F.interpolate(enc4, size=dec4.size()[2:], mode='bilinear', align_corners=False)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4_conv(dec4)
        
        dec3 = self.dec3(dec4)
        if enc3.size()[2:] != dec3.size()[2:]:
            enc3 = F.interpolate(enc3, size=dec3.size()[2:], mode='bilinear', align_corners=False)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3_conv(dec3)
        
        dec2 = self.dec2(dec3)
        if enc2.size()[2:] != dec2.size()[2:]:
            enc2 = F.interpolate(enc2, size=dec2.size()[2:], mode='bilinear', align_corners=False)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2_conv(dec2)
        
        dec1 = self.dec1(dec2)
        if enc1.size()[2:] != dec1.size()[2:]:
            enc1 = F.interpolate(enc1, size=dec1.size()[2:], mode='bilinear', align_corners=False)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1_conv(dec1)
        
        if use_dropout:
            dec1 = self.dropout(dec1)
        
        # Final classification
        out = self.final(dec1)
        
        return out


class PatchDiscriminator(nn.Module):
    """
    PatchGAN discriminator for adversarial training.
    Classifies whether overlapping patches are real or fake.
    """
    def __init__(self, in_channels=1, num_classes=6):
        super(PatchDiscriminator, self).__init__()
        
        # Concatenate input and label map
        self.model = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels + num_classes, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 2
            nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 3
            nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 4
            nn.Conv2d(256, 512, 4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Output layer
            nn.Conv2d(512, 1, 4, stride=1, padding=1)
        )
    
    def forward(self, x, label_map):
        """
        Args:
            x: Input seismic image (B, C, H, W)
            label_map: Label map, either one-hot encoded (B, num_classes, H, W)
                      or class indices (B, H, W)
        """
        # Convert class indices to one-hot if needed
        if label_map.dim() == 3:
            # (B, H, W) -> (B, num_classes, H, W)
            label_map = F.one_hot(label_map.long(), num_classes=6).permute(0, 3, 1, 2).float()
        
        # Concatenate input and label map
        combined = torch.cat([x, label_map], dim=1)
        
        return self.model(combined)


class GANSegmentation(nn.Module):
    """
    Complete GAN-based segmentation model.
    
    Args:
        in_channels: Number of input channels (1 for seismic)
        num_classes: Number of facies classes (6 as per paper)
    """
    def __init__(self, in_channels=1, num_classes=6):
        super(GANSegmentation, self).__init__()
        
        self.generator = UNetGenerator(in_channels, num_classes)
        self.discriminator = PatchDiscriminator(in_channels, num_classes)
        self.num_classes = num_classes
    
    def forward(self, x, use_dropout=False):
        """Forward pass through generator only (for inference)"""
        return self.generator(x, use_dropout)


# ============================================================================
# Model Factory
# ============================================================================

def get_model(model_name='deeplabv3+', in_channels=1, num_classes=6, **kwargs):
    """
    Factory function to create models.
    
    Args:
        model_name: 'deeplabv3+' or 'gan'
        in_channels: Number of input channels
        num_classes: Number of output classes
    
    Returns:
        Model instance
    """
    model_name = model_name.lower()
    
    if model_name == 'deeplabv3+' or model_name == 'deeplab':
        return DeepLabV3Plus(in_channels, num_classes, **kwargs)
    elif model_name == 'gan':
        return GANSegmentation(in_channels, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose 'deeplabv3+' or 'gan'")


if __name__ == "__main__":
    # Test models
    print("Testing DeepLabV3+...")
    model_deeplab = DeepLabV3Plus(in_channels=1, num_classes=6)
    x = torch.randn(2, 1, 200, 200)
    out = model_deeplab(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"✓ DeepLabV3+ test passed!\n")
    
    print("Testing GAN Segmentation...")
    model_gan = GANSegmentation(in_channels=1, num_classes=6)
    out_gen = model_gan(x)
    print(f"Generator output shape: {out_gen.shape}")
    
    # Test discriminator
    labels = torch.randint(0, 6, (2, 200, 200))
    out_disc = model_gan.discriminator(x, labels)
    print(f"Discriminator output shape: {out_disc.shape}")
    print(f"✓ GAN test passed!")
    
    # Count parameters
    print(f"\nDeepLabV3+ parameters: {sum(p.numel() for p in model_deeplab.parameters()):,}")
    print(f"GAN Generator parameters: {sum(p.numel() for p in model_gan.generator.parameters()):,}")
    print(f"GAN Discriminator parameters: {sum(p.numel() for p in model_gan.discriminator.parameters()):,}")
