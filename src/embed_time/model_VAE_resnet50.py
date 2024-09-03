import torch
import torch.nn as  nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels*self.expansion)
        
        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()
        
    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))
        
        x = self.relu(self.batch_norm2(self.conv2(x)))
        
        x = self.conv3(x)
        x = self.batch_norm3(x)
        
        #downsample if needed
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        #add identity
        x+=identity
        x=self.relu(x)
        
        return x

class Block(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Block, self).__init__()
       

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
      identity = x.clone()

      x = self.relu(self.batch_norm2(self.conv1(x)))
      x = self.batch_norm2(self.conv2(x))

      if self.i_downsample is not None:
          identity = self.i_downsample(identity)
      print(x.shape)
      print(identity.shape)
      x += identity
      x = self.relu(x)
      return x


        
        
class ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes, num_channels=3):
        super(ResNet, self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size = 3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*ResBlock.expansion, num_classes)
        
    def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        
        return x
        
    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != planes*ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes*ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes*ResBlock.expansion)
            )
            
        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes*ResBlock.expansion
        
        for i in range(blocks-1):
            layers.append(ResBlock(self.in_channels, planes))
            
        return nn.Sequential(*layers)

        
        
def ResNet50(num_classes, channels=3):
    return ResNet(Bottleneck, [3,4,6,3], num_classes, channels)
    
def ResNet101(num_classes, channels=3):
    return ResNet(Bottleneck, [3,4,23,3], num_classes, channels)

def ResNet152(num_classes, channels=3):
    return ResNet(Bottleneck, [3,8,36,3], num_classes, channels)




# class ResizeConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
#         super().__init__()
#         self.scale_factor = scale_factor
#         self.mode = mode
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size//2)
    
#     def forward(self, x):
#         x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
#         x = self.conv(x)
#         return x

# class BasicBlockEnc(nn.Module):

#     def __init__(self, in_planes, stride=1):
#         super().__init__()

#         planes = in_planes*stride

#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)

#         if stride == 1:
#             self.shortcut = nn.Identity()
#         else:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(planes)
#             )

#     def forward(self, x):
#         out = torch.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out = out + self.shortcut(x)
#         out = torch.relu(out)
#         return out

# class BasicBlockDec(nn.Module):
#     def __init__(self, in_planes, stride=1):
#         super().__init__()
#         planes = int(in_planes/stride)

#         self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(in_planes)
#         # self.bn1 could have been placed here, 
#         # but that messes up the order of the layers when printing the class

#         if stride == 1:
#             self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
#             self.bn1 = nn.BatchNorm2d(planes)
#             self.shortcut = nn.Sequential()
#         else:
#             self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
#             self.bn1 = nn.BatchNorm2d(planes)
#             self.shortcut = nn.Sequential(
#                 ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
#                 nn.BatchNorm2d(planes)
#             )
    
#     def forward(self, x):
#         out = torch.relu(self.bn2(self.conv2(x)))
#         out = self.bn1(self.conv1(out))
#         out = out + self.shortcut(x)
#         out = torch.relu(out)
#         return out


# class ResNet18Enc(nn.Module):

#     def __init__(self, num_Blocks=[2,2,2,2], z_dim=10, nc=3):
#         super().__init__()
#         self.in_planes = 64
#         self.z_dim = z_dim
#         self.conv1 = nn.Conv2d(nc, 64, kernel_size=3, stride=2, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.layer1 = self._make_layer(BasicBlockEnc, 64, num_Blocks[0], stride=1)
#         self.layer2 = self._make_layer(BasicBlockEnc, 128, num_Blocks[1], stride=2)
#         self.layer3 = self._make_layer(BasicBlockEnc, 256, num_Blocks[2], stride=2)
#         self.layer4 = self._make_layer(BasicBlockEnc, 512, num_Blocks[3], stride=2)
#         self.linear = nn.Conv2d(512, 2 * z_dim, kernel_size=1)

#     def _make_layer(self, BasicBlockEnc, planes, num_Blocks, stride):
#         strides = [stride] + [1]*(num_Blocks-1)
#         layers = []
#         for stride in strides:
#             layers += [BasicBlockEnc(self.in_planes, stride)]
#             self.in_planes = planes
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         x = torch.relu(self.bn1(self.conv1(x)))
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = self.linear(x)
#         mu, logvar = torch.chunk(x, 2, dim=1)
#         return mu, logvar

# class ResNet18Dec(nn.Module):

#     def __init__(self, num_Blocks=[2,2,2,2], z_dim=10, nc=3):
#         super().__init__()
#         self.in_planes = 512
#         self.nc = nc

#         self.linear = nn.Conv2d(z_dim, 512, kernel_size=1)

#         self.layer4 = self._make_layer(BasicBlockDec, 256, num_Blocks[3], stride=2)
#         self.layer3 = self._make_layer(BasicBlockDec, 128, num_Blocks[2], stride=2)
#         self.layer2 = self._make_layer(BasicBlockDec, 64, num_Blocks[1], stride=2)
#         self.layer1 = self._make_layer(BasicBlockDec, 64, num_Blocks[0], stride=1)
#         self.conv1 = ResizeConv2d(64, nc, kernel_size=3, scale_factor=2)

#     def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
#         strides = [stride] + [1]*(num_Blocks-1)
#         layers = []
#         for stride in reversed(strides):
#             layers += [BasicBlockDec(self.in_planes, stride)]
#         self.in_planes = planes
#         return nn.Sequential(*layers)

#     def forward(self, z):
#         x = self.linear(z)
#         x = self.layer4(x)
#         x = self.layer3(x)
#         x = self.layer2(x)
#         x = self.layer1(x)
#         x = torch.sigmoid(self.conv1(x))
#         return x
    
    
# class VAEResNet18(nn.Module):
    
#     def __init__(self, nc, z_dim):
#         super().__init__()
#         self.encoder = ResNet18Enc(nc=nc, z_dim=z_dim)
#         self.decoder = ResNet18Dec(nc=nc, z_dim=z_dim)

#     def forward(self, x):
#         mean, logvar = self.encoder(x)
#         z = self.reparameterize(mean, logvar)
#         x = self.decoder(z)
#         return x, z, mean, logvar
    
#     @staticmethod
#     def reparameterize(mean, logvar):
#         std = torch.exp(logvar / 2) # in log-space, squareroot is divide by two
#         epsilon = torch.randn_like(std)
#         return epsilon * std + mean


# class ResizeConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
#         super().__init__()
#         self.scale_factor = scale_factor
#         self.mode = mode
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size//2)
    
#     def forward(self, x):
#         x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
#         x = self.conv(x)
#         return x

# class BasicBlockEnc(nn.Module):

#     def __init__(self, in_planes, stride=1):
#         super().__init__()

#         planes = in_planes*stride

#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)

#         if stride == 1:
#             self.shortcut = nn.Identity()
#         else:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(planes)
#             )

#     def forward(self, x):
#         out = torch.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out = out + self.shortcut(x)
#         out = torch.relu(out)
#         return out

# class BasicBlockDec(nn.Module):
#     def __init__(self, in_planes, stride=1):
#         super().__init__()
#         planes = int(in_planes/stride)

#         self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(in_planes)
#         # self.bn1 could have been placed here, 
#         # but that messes up the order of the layers when printing the class

#         if stride == 1:
#             self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
#             self.bn1 = nn.BatchNorm2d(planes)
#             self.shortcut = nn.Sequential()
#         else:
#             self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
#             self.bn1 = nn.BatchNorm2d(planes)
#             self.shortcut = nn.Sequential(
#                 ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
#                 nn.BatchNorm2d(planes)
#             )
    
#     def forward(self, x):
#         out = torch.relu(self.bn2(self.conv2(x)))
#         out = self.bn1(self.conv1(out))
#         out = out + self.shortcut(x)
#         out = torch.relu(out)
#         return out


# class ResNet18Enc(nn.Module):

#     def __init__(self, num_Blocks=[2,2,2,2], z_dim=10, nc=3):
#         super().__init__()
#         self.in_planes = 64
#         self.z_dim = z_dim
#         self.conv1 = nn.Conv2d(nc, 64, kernel_size=3, stride=2, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.layer1 = self._make_layer(BasicBlockEnc, 64, num_Blocks[0], stride=1)
#         self.layer2 = self._make_layer(BasicBlockEnc, 128, num_Blocks[1], stride=2)
#         self.layer3 = self._make_layer(BasicBlockEnc, 256, num_Blocks[2], stride=2)
#         self.layer4 = self._make_layer(BasicBlockEnc, 512, num_Blocks[3], stride=2)
#         self.linear = nn.Conv2d(512, 2 * z_dim, kernel_size=1)

#     def _make_layer(self, BasicBlockEnc, planes, num_Blocks, stride):
#         strides = [stride] + [1]*(num_Blocks-1)
#         layers = []
#         for stride in strides:
#             layers += [BasicBlockEnc(self.in_planes, stride)]
#             self.in_planes = planes
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         x = torch.relu(self.bn1(self.conv1(x)))
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = self.linear(x)
#         mu, logvar = torch.chunk(x, 2, dim=1)
#         return mu, logvar

# class ResNet18Dec(nn.Module):

#     def __init__(self, num_Blocks=[2,2,2,2], z_dim=10, nc=3):
#         super().__init__()
#         self.in_planes = 512
#         self.nc = nc

#         self.linear = nn.Conv2d(z_dim, 512, kernel_size=1)

#         self.layer4 = self._make_layer(BasicBlockDec, 256, num_Blocks[3], stride=2)
#         self.layer3 = self._make_layer(BasicBlockDec, 128, num_Blocks[2], stride=2)
#         self.layer2 = self._make_layer(BasicBlockDec, 64, num_Blocks[1], stride=2)
#         self.layer1 = self._make_layer(BasicBlockDec, 64, num_Blocks[0], stride=1)
#         self.conv1 = ResizeConv2d(64, nc, kernel_size=3, scale_factor=2)

#     def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
#         strides = [stride] + [1]*(num_Blocks-1)
#         layers = []
#         for stride in reversed(strides):
#             layers += [BasicBlockDec(self.in_planes, stride)]
#         self.in_planes = planes
#         return nn.Sequential(*layers)

#     def forward(self, z):
#         x = self.linear(z)
#         x = self.layer4(x)
#         x = self.layer3(x)
#         x = self.layer2(x)
#         x = self.layer1(x)
#         x = torch.sigmoid(self.conv1(x))
#         return x
    
    
# class VAEResNet18(nn.Module):
    
#     def __init__(self, nc, z_dim):
#         super().__init__()
#         self.encoder = ResNet18Enc(nc=nc, z_dim=z_dim)
#         self.decoder = ResNet18Dec(nc=nc, z_dim=z_dim)

#     def forward(self, x):
#         mean, logvar = self.encoder(x)
#         z = self.reparameterize(mean, logvar)
#         x = self.decoder(z)
#         return x, z, mean, logvar
    
#     @staticmethod
#     def reparameterize(mean, logvar):
#         std = torch.exp(logvar / 2) # in log-space, squareroot is divide by two
#         epsilon = torch.randn_like(std)
#         return epsilon * std + mean
