import torch
import torch.nn as nn
from torchgeometry.image.gaussian import gaussian_blur


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=48, kernel_size=11, stride=4, padding=1)
        self.conv2 = nn.Conv2d(in_channels=48, out_channels=128, kernel_size=5, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=192, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3)
        self.conv5 = nn.Conv2d(in_channels=192, out_channels=128, kernel_size=3, stride=2, padding=1)

        self.batch128 = nn.BatchNorm2d(num_features=128)
        self.batch192 = nn.BatchNorm2d(num_features=192)

        self.full = nn.Linear(in_features=107648, out_features=1024)
        self.out = nn.Linear(in_features=1024, out_features=2)

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

        x = torch.flatten(x, start_dim=1, end_dim=-1)

        x = self.full(x)
        x = self.activation(x)

        x = self.out(x)
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


class Vgg19Bottom(nn.Module):
    def __init__(self, original_model):
        super(Vgg19Bottom, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-5])

    def forward(self, x):
        x = self.features(x)
        return x


class Loss:
    def __init__(self, weight_color, weight_texture, weight_content, weight_tv, gaussian_kernel_shape=21, gaussian_sigma=3):

        self.kernel_size = (gaussian_kernel_shape, gaussian_kernel_shape)
        self.sigma = (gaussian_sigma, gaussian_sigma)

        # loss weights
        self.w_color = weight_color
        self.w_text = weight_texture
        self.w_content = weight_content
        self.w_tv = weight_tv

        vgg19 = torch.hub.load('pytorch/vision:v0.9.0', 'vgg19', pretrained=True)
        self.vgg19 = Vgg19Bottom(vgg19)

    def color_loss(self, edit_image, target_image):
        """color loss - compars colors between edited and target images
        images are blurred to remove textures"""
        blurred_edit = gaussian_blur(edit_image, self.kernel_size, self.sigma)
        blurred_target = gaussian_blur(target_image, self.kernel_size, self.sigma)

        loss = torch.norm(blurred_edit - blurred_target)
        return loss

    def texture_loss(self, edit_image, target_image):
        pass

    def content_loss(self, edit_image, target_image):
        """content loss - compares feature maps to encourage similar features in images"""
        target_vgg = self.vgg19.forward(target_image)
        edit_vgg = self.vgg19.forward(edit_image)

        dist = torch.norm(edit_vgg - target_vgg)

        loss = 1/torch.numel(edit_vgg) * dist

        return loss

    def tv_loss(self, edit_image):
        """total variational loss - enforces spatial smoothness"""
        tv_h = torch.pow(edit_image[:, :, 1:, :] - edit_image[:, :, :-1, :], 2).sum()
        tv_w = torch.pow(edit_image[:, :, :, 1:] - edit_image[:, :, :, :-1], 2).sum()
        return (tv_h + tv_w) / torch.numel(edit_image)

    def total_loss(self, edit_image, target_image):
        color_loss = self.w_color * self.color_loss(edit_image, target_image)
        texture_loss = self.w_text * self.texture_loss(edit_image, target_image)
        content_loss = self.w_content * self.content_loss(edit_image, target_image)
        tv_loss = self.w_tv * self.tv_loss(edit_image)

        return color_loss + texture_loss + content_loss + tv_loss


