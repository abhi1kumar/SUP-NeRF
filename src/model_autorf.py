"""
The encoder follows AutoRF's arch
"""
from typing import Type, Any, Callable, Union, List, Optional
import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1, conv3x3


class ImgEncoder(nn.Module):

    def __init__(self,
                 block: Type[Union[BasicBlock, Bottleneck]],
                 layers: List[int],
                 num_classes: int = 128,
                 zero_init_residual: bool = False,
                 groups: int = 1,
                 width_per_group: int = 64,
                 replace_stride_with_dilation: Optional[List[bool]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4_shape = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.inplanes = 256 * block.expansion  # go back to 256 to use in next _make_layer
        self.layer4_texture = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_shape = nn.Linear(512 * block.expansion, num_classes)
        self.fc_texture = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)  # 1/4
        x = self.layer2(x)  # 1/8
        x = self.layer3(x)  # 1/16

        x_shape = self.layer4_shape(x)  # 1/32
        x_shape = self.avgpool(x_shape)
        x_shape = torch.flatten(x_shape, 1)
        x_shape = self.fc_shape(x_shape)

        x_texture = self.layer4_texture(x)  # 1/32
        x_texture = self.avgpool(x_texture)
        x_texture = torch.flatten(x_texture, 1)
        x_texture = self.fc_texture(x_texture)

        return x_shape, x_texture


def PE(x, degree):
    """
    Positional encoding
    """
    y = torch.cat([2. ** i * x for i in range(degree)], -1)
    w = 1
    return torch.cat([x] + [torch.sin(y) * w, torch.cos(y) * w], -1)


class AutoRF(nn.Module):
    def __init__(self, shape_blocks=5, texture_blocks=5, latent_dim=128,
                 num_xyz_freq=10, num_dir_freq=4, norm_layer_type='BatchNorm2d'):
        super().__init__()
        if norm_layer_type == 'InstanceNorm2d':
            self.img_encoder = ImgEncoder(BasicBlock, [3, 4, 6, 3], num_classes=latent_dim, norm_layer=nn.InstanceNorm2d)
        else:
            self.img_encoder = ImgEncoder(BasicBlock, [3, 4, 6, 3], num_classes=latent_dim, norm_layer=nn.BatchNorm2d)

        self.shape_blocks = shape_blocks
        self.texture_blocks = texture_blocks
        self.num_xyz_freq = num_xyz_freq
        self.num_dir_freq = num_dir_freq

        d_xyz, d_viewdir = 3 + 6 * num_xyz_freq, 3 + 6 * num_dir_freq
        self.encoding_xyz = nn.Sequential(nn.Linear(d_xyz, latent_dim), nn.ReLU())
        for j in range(shape_blocks-1):  # the last block is unique
            layer = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.ReLU())
            setattr(self, f"shape_layer_{j}", layer)
        self.sigma = nn.Sequential(nn.Linear(latent_dim, 1), nn.Softplus())

        # self.encoding_viewdir = nn.Sequential(nn.Linear(d_viewdir, latent_dim), nn.ReLU())
        for j in range(texture_blocks-2):  # the last two block are unique
            layer = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.ReLU())
            setattr(self, f"texture_layer_{j}", layer)
        layer = nn.Sequential(nn.Linear(latent_dim + d_viewdir, latent_dim), nn.ReLU())
        setattr(self, f"texture_layer_{texture_blocks-2}", layer)
        self.rgb = nn.Sequential(nn.Linear(latent_dim + d_viewdir, 3), nn.Sigmoid())

    def encode_img(self, img):
        shape_feat, texture_feat = self.img_encoder(img)
        return shape_feat, texture_feat

    def forward(self, xyz, viewdir, shape_feat, texture_feat):
        xyz = PE(xyz, self.num_xyz_freq)
        pos_feat = self.encoding_xyz(xyz)
        viewdir = PE(viewdir, self.num_dir_freq)

        # Consider xyz concat pixels from a batch of images in dim 0, and shape_feat, texture_feat are from a batch
        bsize = shape_feat.shape[0]
        pixel_per_im = int(xyz.shape[0] / bsize)
        shape_feat = shape_feat.unsqueeze(1).repeat((1, pixel_per_im, 1)).reshape((pixel_per_im * bsize, 1, -1))
        texture_feat = texture_feat.unsqueeze(1).repeat((1, pixel_per_im, 1)).reshape((pixel_per_im * bsize, 1, -1))

        # TODO: really necessary to pass in pos_feat after every layer?
        for j in range(self.shape_blocks-1):
            shape_feat = (shape_feat + pos_feat) / 2
            shape_feat = getattr(self, f"shape_layer_{j}")(shape_feat)
        sigmas = self.sigma((shape_feat + pos_feat) / 2)

        for j in range(self.texture_blocks-2):
            texture_feat = (texture_feat + pos_feat) / 2
            texture_feat = getattr(self, f"texture_layer_{j}")(texture_feat)
        # pass in shape feature
        texture_feat = (texture_feat + shape_feat + pos_feat) / 3
        # TODO: need to pass in view feature in different way?
        texture_feat = torch.cat([texture_feat, viewdir], dim=-1)
        texture_feat = getattr(self, f"texture_layer_{self.texture_blocks-2}")(texture_feat)

        texture_feat = (texture_feat + pos_feat) / 2
        # TODO: need to pass in view feature in different way?
        texture_feat = torch.cat([texture_feat, viewdir], dim=-1)
        rgbs = self.rgb(texture_feat)
        return sigmas, rgbs


# Combined AutoRF encoder and codeNerf decoder
class AutoRFMix(nn.Module):
    def __init__(self, shape_blocks=5, texture_blocks=5, latent_dim=128,
                 num_xyz_freq=10, num_dir_freq=4, norm_layer_type='BatchNorm2d'):
        super().__init__()
        if norm_layer_type == 'InstanceNorm2d':
            self.img_encoder = ImgEncoder(BasicBlock, [3, 4, 6, 3], num_classes=latent_dim, norm_layer=nn.InstanceNorm2d)
        else:
            self.img_encoder = ImgEncoder(BasicBlock, [3, 4, 6, 3], num_classes=latent_dim, norm_layer=nn.BatchNorm2d)

        W = latent_dim
        self.shape_blocks = shape_blocks
        self.texture_blocks = texture_blocks
        self.num_xyz_freq = num_xyz_freq
        self.num_dir_freq = num_dir_freq

        d_xyz, d_viewdir = 3 + 6 * num_xyz_freq, 3 + 6 * num_dir_freq
        self.encoding_xyz = nn.Sequential(nn.Linear(d_xyz, W), nn.ReLU())
        for j in range(shape_blocks):
            layer = nn.Sequential(nn.Linear(latent_dim, W), nn.ReLU())
            setattr(self, f"shape_latent_layer_{j + 1}", layer)
            layer = nn.Sequential(nn.Linear(W, W), nn.ReLU())
            setattr(self, f"shape_layer_{j + 1}", layer)
        self.encoding_shape = nn.Linear(W, W)
        self.sigma = nn.Sequential(nn.Linear(W, 1), nn.Softplus())
        self.encoding_viewdir = nn.Sequential(nn.Linear(W + d_viewdir, W), nn.ReLU())
        for j in range(texture_blocks):
            layer = nn.Sequential(nn.Linear(latent_dim, W), nn.ReLU())
            setattr(self, f"texture_latent_layer_{j + 1}", layer)
            layer = nn.Sequential(nn.Linear(W, W), nn.ReLU())
            setattr(self, f"texture_layer_{j + 1}", layer)
        self.rgb = nn.Sequential(nn.Linear(W, W // 2), nn.ReLU(), nn.Linear(W // 2, 3))

    def encode_img(self, img):
        shape_feat, texture_feat = self.img_encoder(img)
        return shape_feat, texture_feat

    def forward(self, xyz, viewdir, shape_latent, texture_latent):
        xyz = PE(xyz, self.num_xyz_freq)
        viewdir = PE(viewdir, self.num_dir_freq)

        # Consider xyz concat pixels from a batch of images in dim 0, and shape_feat, texture_feat are from a batch
        bsize = shape_latent.shape[0]
        pixel_per_im = int(xyz.shape[0] / bsize)
        shape_latent = shape_latent.unsqueeze(1).repeat((1, pixel_per_im, 1)).reshape((pixel_per_im * bsize, 1, -1))
        texture_latent = texture_latent.unsqueeze(1).repeat((1, pixel_per_im, 1)).reshape((pixel_per_im * bsize, 1, -1))

        y = self.encoding_xyz(xyz)
        for j in range(self.shape_blocks):
            z = getattr(self, f"shape_latent_layer_{j + 1}")(shape_latent)
            y = y + z
            y = getattr(self, f"shape_layer_{j + 1}")(y)
        y = self.encoding_shape(y)
        sigmas = self.sigma(y)
        y = torch.cat([y, viewdir], -1)
        y = self.encoding_viewdir(y)
        for j in range(self.texture_blocks):
            z = getattr(self, f"texture_latent_layer_{j + 1}")(texture_latent)
            y = y + z
            y = getattr(self, f"texture_layer_{j + 1}")(y)
        rgbs = self.rgb(y)
        return sigmas, rgbs


if __name__ == "__main__":
    model = AutoRFMix(latent_dim=256)
    print(model)

    total_params = sum(
        param.numel() for param in model.parameters()
    )
    print(f'total params: {total_params/10**6}')
