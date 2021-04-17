import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=48, kernel_size=11, stride=4)
        self.conv2 = nn.Conv2d(in_channels=48, out_channels=128, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=192, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3)
        self.conv5 = nn.Conv2d(in_channels=192, out_channels=128, kernel_size=3, stride=2)

        self.batch128 = nn.BatchNorm2d(num_features=128)
        self.batch192 = nn.BatchNorm2d(num_features=192)

        self.full = nn.Linear(in_features=128, out_features=1024)

        self.activation = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.activation(x)

        x = self.batch128(x)

        x = self.conv3(x)
        x = self.activation(x)

        x = self.batch192(x)

        x = self.conv4(x)
        x = self.activation(x)

        x = self.batch192(x)

        x = self.conv5(x)
        x = self.activation(x)

        x = self.full(x)

        x = torch.sigmoid(x)

        return x


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.conv9 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)

        self.batch = nn.BatchNorm2d(num_features=64)

        self.activation = nn.ReLU()

        self.tanh = nn.Tanh()

    def residual_block(self, x):
        residual = x

        x = self.conv3(x)
        x = self.activation(x)

        x = self.batch(x)

        x = self.conv3(x)
        x = self.activation(x)

        x = self.batch(x)

        return x + residual

    def forward(self, x):
        x = self.conv9(x)
        x = self.activation(x)

        x = self.residual_block(x)
        x = self.residual_block(x)
        x = self.residual_block(x)
        x = self.residual_block(x)

        x = self.conv3(x)
        x = self.activation(x)

        x = self.conv3(x)
        x = self.activation(x)

        x = self.conv9(x)
        x = self.tanh(x)

        return x


class Loss:
    def __init__(self):
        pass

    def color_loss(self):
        pass

    def texture_loss(self):
        pass

    def content_loss(self):
        pass

    def tv_loss(self):
        pass

    def total_loss(self):
        pass

