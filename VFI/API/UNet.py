import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# -------------------------------- UNet Architecture ---------------------------------- #

# 다운샘플링 유닛 컨볼루션 네트워크. (/2)
# Average Pooling --> Convlution + Leaky ReLU --> Convolution + Leaky ReLU
class down(nn.Module):
    """
    A class for creating neural network blocks containing layers:
    Average Pooling --> Convlution + Leaky ReLU --> Convolution + Leaky ReLU
    This is used in the UNet Class to create a UNet like NN architecture.
    ...

    Methods
    -------
    forward(x)
        Returns output tensor after passing input `x` to the neural network
        block.
    """

    # parms = 입력채널수 / 출력채널수 / 필터크기
    def __init__(self, inChannels, outChannels, filterSize):
        """
        Parameters
        ----------
            inChannels : int
                number of input channels for the first convolutional layer.
            outChannels : int
                number of output channels for the first convolutional layer.
                This is also used as input and output channels for the
                second convolutional layer.
            filterSize : int
                filter size for the convolution filter. input N would create
                a N x N filter.
        """

        super(down, self).__init__()
        # Initialize convolutional layers.
        self.conv1 = nn.Conv2d(inChannels,  outChannels, filterSize, stride=1, padding=int((filterSize - 1) / 2))   # inchannels ----> outchannels
        self.conv2 = nn.Conv2d(outChannels, outChannels, filterSize, stride=1, padding=int((filterSize - 1) / 2))   # outchannels ----> outchaneels
           
    def forward(self, x):
        """
        Returns output tensor after passing input `x` to the neural network
        block.

        Parameters
        ----------
            x : tensor
                input to the NN block.

        Returns
        -------
            tensor
                output of the NN block.
        """

        # Average pooling with kernel size 2 (2 x 2).
        x = F.avg_pool2d(x, 2)
        # Convolution + Leaky ReLU
        x = F.leaky_relu(self.conv1(x), negative_slope = 0.1)
        # Convolution + Leaky ReLU
        x = F.leaky_relu(self.conv2(x), negative_slope = 0.1)
        return x

# 업샘플링 유닛 컨볼루션 네트워크. (*2)
# Bilinear interpolation --> Convlution + Leaky ReLU --> Convolution + Leaky ReLU
class up(nn.Module):
    """
    A class for creating neural network blocks containing layers:
    Bilinear interpolation --> Convlution + Leaky ReLU --> Convolution + Leaky ReLU
    This is used in the UNet Class to create a UNet like NN architecture.
    ...

    Methods
    -------
    forward(x, skpCn)
        Returns output tensor after passing input `x` to the neural network
        block.
    """

    # parms = 입력채널수 / 출력채널수
    def __init__(self, inChannels, outChannels):
        """
        Parameters
        ----------
            inChannels : int
                number of input channels for the first convolutional layer.
            outChannels : int
                number of output channels for the first convolutional layer.
                This is also used for setting input and output channels for
                the second convolutional layer.
        """

        # 필터 크기가 3으로 고정되며, 잔차 연결을 위한 입력채널의 2배수 적용.
        super(up, self).__init__()
        # Initialize convolutional layers.
        self.conv1 = nn.Conv2d(inChannels,  outChannels, 3, stride=1, padding=1)
        # (2 * outChannels) is used for accommodating skip connection.
        self.conv2 = nn.Conv2d(2 * outChannels, outChannels, 3, stride=1, padding=1)
    
    # params = 입력레이어 / 잔차연결레이어
    def forward(self, x, skpCn):
        """
        Returns output tensor after passing input `x` to the neural network
        block.

        Parameters
        ----------
            x : tensor
                input to the NN block.
            skpCn : tensor
                skip connection input to the NN block.

        Returns
        -------
            tensor
                output of the NN block.
        """

        # Bilinear interpolation with scaling 2.
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        # Convolution + Leaky ReLU
        x = F.leaky_relu(self.conv1(x), negative_slope = 0.1)                               # inChannels ----> outChannels
        # Convolution + Leaky ReLU on (`x`, `skpCn`)
        x = F.leaky_relu(self.conv2(torch.cat((x, skpCn), 1)), negative_slope = 0.1)        # outChannels + skpCn(outch) ----> outChannels
        return x

# UNet Network combine UP & Down Class
class UNet(nn.Module):
    """
    A class for creating UNet like architecture as specified by the
    Super SloMo paper.
    ...

    Methods
    -------
    forward(x)
        Returns output tensor after passing input `x` to the neural network
        block.
    """

    # params = 입력채널수 / 출력채널수
    def __init__(self, inChannels, outChannels):
        """
        Parameters
        ----------
            inChannels : int
                number of input channels for the UNet.
            outChannels : int
                number of output channels for the UNet.
        """

        # 네트워크 구조 : 공간 정보 획득 2 layer / down 5 layer / up 6 layer / Output layer
        super(UNet, self).__init__()

        # Initialize neural network blocks.
        self.conv1 = nn.Conv2d(inChannels, 32, 7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(32, 32, 7, stride=1, padding=3)
        self.down1 = down(32, 64, 5)
        self.down2 = down(64, 128, 3)
        self.down3 = down(128, 256, 3)
        self.down4 = down(256, 512, 3)
        self.up1   = up(512, 256)
        self.up2   = up(256, 128)
        self.up3   = up(128, 64)
        self.up4   = up(64, 32)
        self.conv3 = nn.Conv2d(32, outChannels, 3, stride=1, padding=1)
        
    
    # UNet Forward - Input of UNet -> Output of UNet
    def forward(self, x):
        """
        Returns output tensor after passing input `x` to the neural network.

        Parameters
        ----------
            x : tensor
                input to the UNet.

        Returns
        -------
            tensor
                output of the UNet.
        """

        # 공간정보획득 : (Conv + Leaky_relu) * 2 
        x  = F.leaky_relu(self.conv1(x), negative_slope = 0.1)
        s1 = F.leaky_relu(self.conv2(x), negative_slope = 0.1)

        # Encoder(down) : [AvgPool + (Conv + Leaky_relu) * 2] * 5
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        x = self.down4(s4)

        # Decoder(up) : [bi-interpolation + (Conv + Leaky_relu) * 2] * 5
        x  = self.up1(x, s4)
        x  = self.up2(x, s3)
        x  = self.up3(x, s2)
        x  = self.up4(x, s1)

        # Output Layer : Conv + Leaky_relu
        x  = F.leaky_relu(self.conv3(x), negative_slope = 0.1)
        return x