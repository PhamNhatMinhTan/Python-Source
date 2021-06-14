import torch
import torch.nn as nn
import torch.nn.functional as F


class conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, bias):
        super(conv2DBatchNormRelu, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)  # Conv2D
        self.batchnorm = nn.BatchNorm2d(out_channels)  # BN (batchnorm)
        # ReLU
        self.relu = nn.ReLU(inplace=True)  # Set implace = true to value process in memory

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        outputs = self.relu(x)

        return outputs


### Feature Module (Encoder) ###
class FeatureMap_convolution(nn.Module):
    def __init__(self):
        super(FeatureMap_convolution, self).__init__()

        # Block 1
        in_channels, out_channels, kernel_size, stride, padding, dilation, bias = 3, 64, 3, 2, 1, 1, False
        self.cbnr_1 = conv2DBatchNormRelu(in_channels, out_channels, kernel_size, stride, padding, dilation, bias)

        # Block 2
        in_channels, out_channels, kernel_size, stride, padding, dilation, bias = 64, 64, 3, 1, 1, 1, False
        self.cbnr_2 = conv2DBatchNormRelu(in_channels, out_channels, kernel_size, stride, padding, dilation, bias)

        # Block 3
        in_channels, out_channels, kernel_size, stride, padding, dilation, bias = 64, 128, 3, 1, 1, 1, False
        self.cbnr_3 = conv2DBatchNormRelu(in_channels, out_channels, kernel_size, stride, padding, dilation, bias)

        # Block 4: Max pooling
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.cbnr_1(x)
        x = self.cbnr_2(x)
        x = self.cbnr_3(x)
        outputs = self.maxpool(x)

        return outputs


class ResidualBlockPSP(nn.Sequential):
    def __init__(self, n_blocks, in_channels, mid_channels, out_channels, stride, dilation):
        super().__init__()  # The syntax super() method of python 2

        # BottleNeckPSP
        self.add_module("block1", BottleNeckPSP(in_channels, mid_channels, out_channels, stride, dilation))

        # BottleNeckIdentifyPSP
        for i in range(n_blocks - 1):
            self.add_module("block" + str(i + 2), BottleNeckIdentifyPSP(out_channels, mid_channels, stride, dilation))


class conv2DBatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, bias):
        super().__init__()  # The syntax super() method of python 3
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)  # C3
        self.batchnorm = nn.BatchNorm2d(out_channels)  # BN (batchnorm)

    def forward(self, x):
        x = self.conv(x)
        outputs = self.batchnorm(x)

        return outputs


class BottleNeckPSP(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride, dilation):
        super().__init__()  # This does the same thing as:  super(BottleNeckPSP, self).__init__()

        # Block 1
        self.cbnr_1 = conv2DBatchNormRelu(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

        # Block 2
        self.cbnr_2 = conv2DBatchNormRelu(
            mid_channels, mid_channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False
        )

        # Block 3
        self.cbn_3 = conv2DBatchNorm(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

        ### SKIP CONNECTION ###
        # Block 4
        self.cbn_residual = conv2DBatchNorm(
            in_channels, out_channels, kernel_size=1, stride=stride, padding=0, dilation=1, bias=False
        )

        # ReLU
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.cbn_3(self.cbnr_2(self.cbnr_1(x)))
        residual = self.cbn_residual(x)

        return self.relu(conv + residual)


class BottleNeckIdentifyPSP(nn.Module):
    def __init__(self, in_channels, mid_channels, stride, dilation):
        super().__init__()

        # Block 1
        self.cbnr_1 = conv2DBatchNormRelu(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

        # Block 2
        self.cbnr_2 = conv2DBatchNormRelu(
            mid_channels, mid_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False
        )

        # Block 3
        self.cbn_3 = conv2DBatchNorm(mid_channels, in_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

        # ReLU
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.cbn_3(self.cbnr_2(self.cbnr_1(x)))
        residual = x

        return self.relu(conv + residual)


### Payramid Pooling Module ###
class PyramidPooling(nn.Module):
    def __init__(self, in_channels, pool_sizes, height, width):
        super().__init__()

        self.height = height
        self.width = width

        out_channels = int(in_channels / len(pool_sizes))
        # pool_sizes = [6, 3, 2, 1]
        # P6
        self.avgpool_1 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[0])
        self.cbnr_1 = conv2DBatchNormRelu(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

        # P3
        self.avgpool_2 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[1])
        self.cbnr_2 = conv2DBatchNormRelu(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

        # P2
        self.avgpool_3 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[2])
        self.cbnr_3 = conv2DBatchNormRelu(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

        # P1
        self.avgpool_4 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[3])
        self.cbnr_4 = conv2DBatchNormRelu(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

    def forward(self, x):
        # P6
        out1 = self.cbnr_1(self.avgpool_1(x))
        out1 = F.interpolate(
            out1,
            size=(self.height, self.width),
            mode="bilinear",
            align_corners=True,
        )

        # P3
        out2 = self.cbnr_2(self.avgpool_2(x))
        out2 = F.interpolate(out2, size=(self.height, self.width), mode="bilinear", align_corners=True)

        # P2
        out3 = self.cbnr_3(self.avgpool_3(x))
        out3 = F.interpolate(out3, size=(self.height, self.width), mode="bilinear", align_corners=True)

        # P1
        out4 = self.cbnr_4(self.avgpool_4(x))
        out4 = F.interpolate(out4, size=(self.height, self.width), mode="bilinear", align_corners=True)

        output = torch.cat([x, out1, out2, out3, out4], dim=1)

        return output


### Decoder Module (Upsampling) ###
class DecodePSPFeature(nn.Module):
    def __init__(self, height, width, n_classes):
        super().__init__()

        self.height = height
        self.width = width

        self.cbnr = conv2DBatchNormRelu(
            in_channels=4096, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=False
        )
        self.dropout = nn.Dropout(p=0.1)  # Random cut 10% nodes in the layer
        self.classification = nn.Conv2d(in_channels=512, out_channels=n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.cbnr(x)
        x = self.dropout(x)
        x = self.classification(x)
        output = F.interpolate(x, size=(self.height, self.width), mode="bilinear", align_corners=True)

        return output


### AuxLoss Module ###
class AuxiliaryPSPLayers(nn.Module):
    def __init__(self, in_channels, height, width, n_classes):
        super().__init__()

        self.height = height
        self.width = width

        self.cbnr = conv2DBatchNormRelu(
            in_channels=in_channels, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, bias=False
        )
        self.dropout = nn.Dropout(p=0.1)  # Random cut 10% nodes in the layer
        self.classification = nn.Conv2d(in_channels=256, out_channels=n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.cbnr(x)
        x = self.dropout(x)
        x = self.classification(x)
        output = F.interpolate(x, size=(self.height, self.width), mode="bilinear", align_corners=True)

        return output


### PSPNet ###
class PSPNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        # Parameters
        block_config = [3, 4, 6, 3]
        img_size = 475
        img_size_8 = 60

        # Feature module
        self.feature_convolution = FeatureMap_convolution()
        self.feature_residualBlock_1 = ResidualBlockPSP(
            n_blocks=block_config[0], in_channels=128, mid_channels=64, out_channels=256, stride=1, dilation=1
        )
        self.feature_residualBlock_2 = ResidualBlockPSP(
            n_blocks=block_config[1], in_channels=256, mid_channels=128, out_channels=512, stride=2, dilation=1
        )
        self.feature_dilated_residualBlock_1 = ResidualBlockPSP(
            n_blocks=block_config[2], in_channels=512, mid_channels=256, out_channels=1024, stride=1, dilation=2
        )
        self.feature_dilated_residualBlock_2 = ResidualBlockPSP(
            n_blocks=block_config[3], in_channels=1024, mid_channels=512, out_channels=2048, stride=1, dilation=4
        )

        # Pyramid Pooling module
        self.pyramid_pooling = PyramidPooling(in_channels=2048, pool_sizes=[6, 3, 2, 1], height=img_size_8, width=img_size_8)

        # Decoder module
        self.decoder_feature = DecodePSPFeature(height=img_size, width=img_size, n_classes=n_classes)

        # AuxLoss module
        self.auxLoss = AuxiliaryPSPLayers(in_channels=1024, height=img_size, width=img_size, n_classes=n_classes)

    def forward(self, x):
        x = self.feature_convolution(x)
        x = self.feature_residualBlock_1(x)
        x = self.feature_residualBlock_2(x)
        x = self.feature_dilated_residualBlock_1(x)
        output_auxLoss = self.auxLoss(x)
        x = self.feature_dilated_residualBlock_2(x)
        x = self.pyramid_pooling(x)
        output = self.decoder_feature(x)

        return (output, output_auxLoss)


if __name__ == "__main__":
    # x = torch.randn(1, 3, 475, 475)

    dummy_img = torch.rand(2, 3, 475, 475)
    net = PSPNet(21)
    outputs = net(dummy_img)
    print(outputs[0].shape)
