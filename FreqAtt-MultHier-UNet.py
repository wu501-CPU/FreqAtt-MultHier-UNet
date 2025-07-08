import torch
import torch.nn as nn
import torch.nn.functional as F
from math import gcd


class FreqAttMultHierUNet(nn.Module):
    """FreqAtt-MultHier-UNet: Frequency Attention Multi-Hierarchical UNet"""

    def __init__(self, in_channels=3, base_channels=64, num_classes=1):
        super(FreqAttMultHierUNet, self).__init__()

        # Encoder stages
        self.encoder = Encoder(in_channels, base_channels)

        # Decoder
        decoder_channels = [base_channels * 4, base_channels * 2, base_channels, base_channels // 2]
        self.decoder = HMFFDecoder([base_channels, base_channels * 2, base_channels * 4, base_channels * 8],
                                   decoder_channels)

        # Final prediction
        self.final_conv = nn.Conv2d(decoder_channels[-1], num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder pathway
        enc_features = self.encoder(x)

        # Decoder pathway
        dec_out = self.decoder(*enc_features)

        # Final prediction
        out = self.final_conv(dec_out)

        return torch.sigmoid(out)


class Encoder(nn.Module):
    """Encoder with Dual-Frequency Blocks and downsampling"""

    def __init__(self, in_channels, base_channels):
        super(Encoder, self).__init__()

        # Initial convolution
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # Encoder stages
        self.stage1 = EncoderStage(base_channels, base_channels, alpha_in=0.0)
        self.stage2 = EncoderStage(base_channels, base_channels * 2)
        self.stage3 = EncoderStage(base_channels * 2, base_channels * 4)
        self.stage4 = EncoderStage(base_channels * 4, base_channels * 8)

        # Multi-Scale Context Aggregation in bottleneck
        self.bottleneck = nn.Sequential(
            MultiScaleDualAttentionFusionBlock(base_channels * 8),
            MultiScaleAttentionGate(base_channels * 8),
            MultiScaleDualAttentionFusionBlock(base_channels * 8)
        )

    def forward(self, x):
        features = []

        # Initial convolution
        x = self.init_conv(x)

        # Stage 1
        x = self.stage1(x)
        features.append(x)

        # Stage 2
        x = self.stage2(x)
        features.append(x)

        # Stage 3
        x = self.stage3(x)
        features.append(x)

        # Stage 4
        x = self.stage4(x)

        # Bottleneck
        x = self.bottleneck(x)
        features.append(x)

        return features


class EncoderStage(nn.Module):
    """Encoder stage with Dual-Frequency Block and downsampling"""

    def __init__(self, in_channels, out_channels, alpha_in=None):
        super(EncoderStage, self).__init__()

        self.dfb = DualFrequencyBlock(in_channels, out_channels, alpha_in=alpha_in)
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        self.msag = MultiScaleAttentionGate(out_channels)

    def forward(self, x):
        x = self.dfb(x)
        x = self.msag(x)
        x = self.downsample(x)
        return x


class DualFrequencyBlock(nn.Module):
    def __init__(self, in_channels, out_channels, alpha=0.5, alpha_in=None, alpha_out=None):
        super(DualFrequencyBlock, self).__init__()

        self.alpha = alpha
        self.alpha_in = alpha_in if alpha_in is not None else alpha
        self.alpha_out = alpha_out if alpha_out is not None else alpha

        # Calculate channel numbers for high and low frequency components
        self.in_channels_h = int(in_channels * (1 - self.alpha_in))
        self.in_channels_l = in_channels - self.in_channels_h

        self.out_channels_h = int(out_channels * (1 - self.alpha_out))
        self.out_channels_l = out_channels - self.out_channels_h

        # 1) Frequency Feature Decoupling is handled in forward pass

        # 2) Four-path Cross-frequency Interaction
        # High-frequency preserving path (X_h -> X_hh)
        self.conv_hh = nn.Conv2d(self.in_channels_h, self.out_channels_h,
                                 kernel_size=3, padding=1)

        # High-to-low frequency migration path (X_h -> X_hl)
        self.avgpool_h = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv_hl = nn.Conv2d(self.in_channels_h, self.out_channels_l,
                                 kernel_size=3, padding=1)

        # Low-frequency preserving path (X_l -> X_ll)
        self.conv_ll = nn.Conv2d(self.in_channels_l, self.out_channels_l,
                                 kernel_size=3, padding=2, dilation=2)

        # Low-to-high frequency migration path (X_l -> X_lh)
        self.conv_lh = nn.Conv2d(self.in_channels_l, self.out_channels_h,
                                 kernel_size=3, padding=1)
        self.upsample_lh = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # 4) Dynamic Channel Calibration
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # 1) Frequency Feature Decoupling
        # Split input channels into high and low frequency components
        x_h = x[:, :self.in_channels_h, :, :]  # High-frequency component
        x_l = self.avgpool_h(x[:, self.in_channels_h:, :, :])  # Low-frequency component

        # 2) Four-path Cross-frequency Interaction
        # High-frequency preserving path
        x_hh = self.conv_hh(x_h)

        # High-to-low frequency migration path
        x_hp = self.avgpool_h(x_h)
        x_hl = self.conv_hl(x_hp)

        # Low-frequency preserving path
        x_ll = self.conv_ll(x_l)

        # Low-to-high frequency migration path
        x_lh = self.conv_lh(x_l)
        x_lh = self.upsample_lh(x_lh)

        # 3) Cross-frequency Feature Fusion
        # High-frequency branch fusion
        x_h_prime = x_hh + x_lh

        # Low-frequency branch fusion
        x_hl_pooled = self.avgpool_h(x_hl) if x_hl.size(2) > x_ll.size(2) else x_hl
        x_l_prime = x_ll + x_hl_pooled

        # 4) Dynamic Channel Calibration
        # Get channel descriptors
        s1 = self.global_pool(x_h_prime)
        s2 = self.global_pool(x_l_prime)

        # Concatenate and get attention weights
        s = torch.cat([s1, s2], dim=1)
        beta = self.softmax(s)
        beta1, beta2 = torch.split(beta, [self.out_channels_h, self.out_channels_l], dim=1)

        # Apply channel-wise attention
        x_h_attended = x_h_prime * beta1.expand_as(x_h_prime)
        x_l_attended = x_l_prime * beta2.expand_as(x_l_prime)

        # Combine features
        # Upsample low-frequency features if needed
        if x_l_attended.size(2) < x_h_attended.size(2):
            x_l_attended = F.interpolate(x_l_attended, size=x_h_attended.size()[2:],
                                         mode='bilinear', align_corners=True)

        # Concatenate along channel dimension
        y = torch.cat([x_h_attended, x_l_attended], dim=1)

        return y


class ChannelAttentionBlock(nn.Module):
    """ Channel Attention Block (CAB) """

    def __init__(self, channels, reduction_ratio=16):
        super(ChannelAttentionBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        mid_channels = channels // reduction_ratio
        self.conv1 = nn.Conv2d(channels, mid_channels, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(mid_channels, channels, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Dual pooling paths
        avg_out = self.conv2(self.relu(self.conv1(self.avg_pool(x))))
        max_out = self.conv2(self.relu(self.conv1(self.max_pool(x))))

        # Feature fusion and attention generation
        channel_att = self.sigmoid(avg_out + max_out)
        return x * channel_att


class SpatialAttentionBlock(nn.Module):
    """ Spatial Attention Block (SAB) """

    def __init__(self, kernel_sizes=[3, 7, 11]):
        super(SpatialAttentionBlock, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(2, 1, kernel_size=k, padding=k // 2, bias=False)
            for k in kernel_sizes
        ])
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel statistics
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)

        # Multi-scale spatial attention
        attn_maps = []
        for conv in self.convs:
            attn_maps.append(conv(combined))
        spatial_att = self.sigmoid(torch.mean(torch.stack(attn_maps), dim=0))
        return x * spatial_att


class MultiScaleParallelDWConv(nn.Module):
    """ Multi-Scale Parallel Depthwise Convolution (MSPDC) """

    def __init__(self, channels, kernel_sizes=[1, 3, 5]):
        super(MultiScaleParallelDWConv, self).__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=k,
                          padding=k // 2, groups=channels, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            ) for k in kernel_sizes
        ])

    def channel_shuffle(self, x, groups):
        batch, channels, height, width = x.size()
        channels_per_group = channels // groups
        x = x.view(batch, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        return x.view(batch, channels, height, width)

    def forward(self, x, mode='parallel'):
        if mode == 'parallel':
            # Parallel multi-scale fusion
            features = [conv(x) for conv in self.convs]
            out = torch.cat(features, dim=1)
            out = self.channel_shuffle(out, gcd(out.size(1), x.size(1)))
        else:
            # Serial residual learning
            out = x
            for conv in self.convs:
                out = out + conv(out)
        return out


class MultiScaleDualAttentionFusionBlock(nn.Module):
    """ Multi-Scale Dual-Attention Fusion Block (MSDAFB) """

    def __init__(self, channels):
        super(MultiScaleDualAttentionFusionBlock, self).__init__()
        self.cab = ChannelAttentionBlock(channels)
        self.sab = SpatialAttentionBlock()
        self.mspdc = MultiScaleParallelDWConv(channels)
        self.project = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        # Dual-attention feature refinement
        x = self.cab(x)
        x = self.sab(x)

        # Multi-scale feature extraction
        x = self.mspdc(x)
        x = self.project(x)
        return x


class EfficientUpsamplingConvBlock(nn.Module):
    """ Efficient Upsampling Convolution Block (EUCB) """

    def __init__(self, in_channels, out_channels):
        super(EfficientUpsamplingConvBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dw_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3,
                      padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.pw_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.upsample(x)
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x


class ConvMixerBlock(nn.Module):
    """ ConvMixer Block """

    def __init__(self, channels, kernel_size=7, depth=2):
        super(ConvMixerBlock, self).__init__()
        self.blocks = nn.Sequential(*[
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=kernel_size,
                          groups=channels, padding=kernel_size // 2, bias=False),
                nn.BatchNorm2d(channels),
                nn.GELU(),
                nn.Conv2d(channels, channels, kernel_size=1, bias=True),
                nn.BatchNorm2d(channels),
                nn.GELU()
            ) for _ in range(depth)
        ])

    def forward(self, x):
        return self.blocks(x)


class MultiScaleAttentionGate(nn.Module):
    """ Multi-Scale Attention Gate (MSAG) """

    def __init__(self, channels):
        super(MultiScaleAttentionGate, self).__init__()
        self.pointwise = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.ordinary = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.dilated = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3,
                      padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        f1 = self.pointwise(x)
        f2 = self.ordinary(x)
        f3 = self.dilated(x)
        combined = torch.cat([f1, f2, f3], dim=1)
        attn = self.fusion(F.relu(combined))
        return x + x * attn


class HMFFDecoder(nn.Module):
    """ Hierarchical Multi-scale Feature Fusion Decoder """

    def __init__(self, encoder_channels, decoder_channels=[256, 128, 64, 32]):
        super(HMFFDecoder, self).__init__()
        self.decoder_blocks = nn.ModuleList()
        self.attention_gates = nn.ModuleList()

        # Reverse encoder channels for decoder
        encoder_channels = encoder_channels[::-1]

        for i in range(len(decoder_channels)):
            in_ch = encoder_channels[i] if i == 0 else encoder_channels[i] + decoder_channels[i - 1]
            out_ch = decoder_channels[i]

            self.decoder_blocks.append(nn.Sequential(
                ConvMixerBlock(in_ch),
                MultiScaleDualAttentionFusionBlock(in_ch),
                EfficientUpsamplingConvBlock(in_ch, out_ch)
            ))

            self.attention_gates.append(MultiScaleAttentionGate(out_ch))

    def forward(self, *features):
        features = features[::-1]  # Reverse encoder features

        x = features[0]
        decoder_outputs = []

        for i, (dec_block, attn_gate) in enumerate(zip(self.decoder_blocks, self.attention_gates)):
            # Process through decoder block
            x = dec_block(x)

            # Skip connection from encoder
            if i < len(features) - 1:
                skip = features[i + 1]
                # Ensure skip connection has matching spatial dimensions
                if skip.size(2) != x.size(2) or skip.size(3) != x.size(3):
                    skip = F.interpolate(skip, size=x.size()[2:], mode='bilinear', align_corners=True)
                x = torch.cat([x, skip], dim=1)

            # Apply attention gate
            x = attn_gate(x)
            decoder_outputs.append(x)

        return decoder_outputs[-1]