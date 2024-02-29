import torch
import torch.nn as nn
import torch.nn.functional as F

# AlexNet
class AlexNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
       super().__init__()
       self.in_channels = in_channels
       self.num_classes = num_classes
       self.net = nn.Sequential(
                    # 这里使用一个11*11的更大窗口来捕捉对象。
                    # 同时，步幅为4，以减少输出的高度和宽度。
                    # 另外，输出通道的数目远大于LeNet
                    nn.Conv2d(self.in_channels, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                    # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
                    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                    # 使用三个连续的卷积层和较小的卷积窗口。
                    # 除了最后的卷积层，输出通道的数量进一步增加。
                    # 在前两个卷积层之后，汇聚层不用于减少输入的高度和宽度
                    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
                    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
                    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                    nn.Flatten(),
                    # 这里，全连接层的输出数量是LeNet中的好几倍。使用dropout层来减轻过拟合
                    nn.Linear(6400, 4096), nn.ReLU(),
                    nn.Dropout(p=0.5),
                    nn.Linear(4096, 4096), nn.ReLU(),
                    nn.Dropout(p=0.5),
                    # 最后是输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
                    nn.Linear(4096, self.num_classes))
    
    def forward(self, x):
        return self.net(x)
 
# LeNet
class LeNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=10) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.net = nn.Sequential(nn.Conv2d(self.in_channels, 6, kernel_size=5, padding=2), 
                              nn.Sigmoid(),
                              nn.AvgPool2d(kernel_size=2, stride=2),
                              nn.Conv2d(6, 16, kernel_size=5), 
                              nn.Sigmoid(),
                              nn.AvgPool2d(kernel_size=2, stride=2),
                              nn.Flatten(),
                              nn.Linear(16 * 5 * 5, 120), 
                              nn.Sigmoid(),
                              nn.Linear(120, 84),
                              nn.Sigmoid(),
                              nn.Linear(84, self.num_classes))
        
    def forward(self, x):
        return self.net(x)

# LeNet_BN
class LeNet_BN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.net = nn.Sequential(nn.Conv2d(self.in_channels, 6, kernel_size=5, padding=2), 
                              nn.BatchNorm2d(6),
                              nn.Sigmoid(),
                              nn.AvgPool2d(kernel_size=2, stride=2),
                              nn.Conv2d(6, 16, kernel_size=5), 
                              nn.BatchNorm2d(16),
                              nn.Sigmoid(),
                              nn.AvgPool2d(kernel_size=2, stride=2),
                              nn.Flatten(),
                              nn.Linear(16 * 5 * 5, 120), 
                              nn.Sigmoid(),
                              nn.Linear(120, 84),
                              nn.Sigmoid(),
                              nn.Linear(84, self.num_classes))
        
    def forward(self, x):
        return self.net(x)


# LeNet_BN2
class LeNet_BN2(nn.Module):
    def __init__(self, in_channels=1, num_classes=10) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.net = nn.Sequential(nn.Conv2d(self.in_channels, 6, kernel_size=5, padding=2), 
                              nn.BatchNorm2d(6),
                              nn.Sigmoid(),
                              nn.AvgPool2d(kernel_size=2, stride=2),
                              nn.Conv2d(6, 16, kernel_size=5), 
                              nn.BatchNorm2d(16),
                              nn.Sigmoid(),
                              nn.AvgPool2d(kernel_size=2, stride=2),
                              nn.Flatten(),
                              nn.Linear(16 * 5 * 5, 120), 
                              nn.BatchNorm1d(120),
                              nn.Sigmoid(),
                              nn.Linear(120, 84),
                              nn.BatchNorm1d(84),
                              nn.Sigmoid(),
                              nn.Linear(84, self.num_classes))
        
    def forward(self, x):
        return self.net(x)

# LeNet_all_BN
class LeNet_all_BN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.net = nn.Sequential(nn.Conv2d(self.in_channels, 6, kernel_size=5, padding=2),  
                                 nn.BatchNorm2d(6),
                                 nn.Sigmoid(),
                                 nn.BatchNorm2d(6),
                                 nn.AvgPool2d(kernel_size=2, stride=2),
                                 nn.BatchNorm2d(6),
                                 nn.Conv2d(6, 16, kernel_size=5),
                                 nn.BatchNorm2d(16),
                                 nn.Sigmoid(),
                                 nn.BatchNorm2d(16),
                                 nn.AvgPool2d(kernel_size=2, stride=2),
                                 nn.BatchNorm2d(16),
                                 nn.Flatten(),
                                 nn.Linear(16 * 5 * 5, 120),
                                 nn.Sigmoid(),
                                 nn.BatchNorm1d(120),
                                 nn.Linear(120, 84),
                                 nn.BatchNorm1d(84),
                                 nn.Sigmoid(),
                                 nn.BatchNorm1d(84),
                                 nn.Linear(84, 10),
                                 nn.BatchNorm1d(10)
                                 )
        
    def forward(self, x):
        return self.net(x) 


# diy Batch_Norm layer to replace nn.BatchNorm2d 
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    
    # 通过is_grad_enabled来判断当前模式是训练模式还是预测模式
    if not torch.is_grad_enabled():
        # 如果是在预测模式下，直接使用传入的移动平均所得的均值和方差
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        # 限制X的维度
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # 使用全连接层的情况，计算特征维上的均值和方差
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # 使用二维卷积层的情况，计算通道维上（axis=1）的均值和方差。
            # 这里我们需要保持X的形状以便后面可以做广播运算
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        # 训练模式下，用当前的均值和方差做标准化
        X_hat = (X - mean) / torch.sqrt(var + eps)
        
        # 更新移动平均的均值和方差
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    
    Y = gamma * X_hat + beta  # 缩放和移位
    return Y, moving_mean.data, moving_var.data

class BatchNorm(nn.Module):
    # num_features：完全连接层的输出数量或卷积层的输出通道数。
    
    # num_dims：2表示完全连接层，4表示卷积层
    def __init__(self, num_features, num_dims, freeze=False):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # 参与求梯度和迭代的拉伸和偏移参数，分别初始化成1和0
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # 非模型参数的变量初始化为0和1
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)
        self.freeze = freeze
            
        

    def forward(self, X):
        if self.freeze:
            m = 1.0
        else:
            m = 0.9
        # 如果X不在显存上，将moving_mean和moving_var复制到X所在显存上
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # 保存更新过的moving_mean和moving_var
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=m)
        return Y

class LeNet_BN_freeze(nn.Module):
    def __init__(self, in_channels=1, num_classes=10) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.net = nn.Sequential(nn.Conv2d(self.in_channels, 6, kernel_size=5, padding=2), 
                              BatchNorm(6, 4, freeze=True),
                              nn.Sigmoid(),
                              nn.AvgPool2d(kernel_size=2, stride=2),
                              nn.Conv2d(6, 16, kernel_size=5), 
                              BatchNorm(16, 4, freeze=True),
                              nn.Sigmoid(),
                              nn.AvgPool2d(kernel_size=2, stride=2),
                              nn.Flatten(),
                              nn.Linear(16 * 5 * 5, 120), 
                              nn.Sigmoid(),
                              nn.Linear(120, 84),
                              nn.Sigmoid(),
                              nn.Linear(84, self.num_classes))
    def forward(self, x):
        return self.net(x)



class LeNet_BN2_freeze(nn.Module):
    def __init__(self, in_channels=1, num_classes=10) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.net = nn.Sequential(nn.Conv2d(self.in_channels, 6, kernel_size=5, padding=2), 
                              BatchNorm(6, 4, freeze=True),
                              nn.Sigmoid(),
                              nn.AvgPool2d(kernel_size=2, stride=2),
                              nn.Conv2d(6, 16, kernel_size=5), 
                              BatchNorm(16, 4, freeze=True),
                              nn.Sigmoid(),
                              nn.AvgPool2d(kernel_size=2, stride=2),
                              nn.Flatten(),
                              nn.Linear(16 * 5 * 5, 120), 
                              BatchNorm(120, 2, freeze=True),
                              nn.Sigmoid(),
                              nn.Linear(120, 84),
                              BatchNorm(84, 2, freeze=True),
                              nn.Sigmoid(),
                              nn.Linear(84, self.num_classes))
    def forward(self, x):
        return self.net(x)



# LeNetPro: use ReLU instead of Sigmoid; append dropout layer after linear layer
class LeNetPro(nn.Module):
    def __init__(self, in_channels=1, num_classes=10) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.net = nn.Sequential(nn.Conv2d(self.in_channels, 6, kernel_size=5, padding=2),
                                 nn.ReLU(),
                                 nn.AvgPool2d(kernel_size=2, stride=2),
                                 nn.Conv2d(6, 16, kernel_size=5), 
                                 nn.ReLU(),
                                 nn.AvgPool2d(kernel_size=2, stride=2),
                                 nn.Flatten(),
                                 nn.Linear(16 * 5 * 5, 120), 
                                 nn.ReLU(),
                                 nn.Dropout(p=0.5),
                                 nn.Linear(120, 84),
                                 nn.ReLU(),         
                                 nn.Dropout(p=0.5),
                                 nn.Linear(84, self.num_classes))
    def forward(self, x):
        return self.net(x)


# idea: to simplify the AlexNet
class AlexNetSimple(nn.Module):
    def __init__(self, in_channels=1, num_classes=10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            nn.Flatten(),
            
            nn.Linear(256 * 6 * 6, 2048),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)
    
# net = AlexNetSimple(1, 10).cuda()
# # import torchsummary
# # torchsummary.summary(net, (3, 224, 224))
# # net = AlexNet()
# # print(net)
# x= torch.randn(1, 1, 28, 28).cuda()
# print(net(x).shape)


# VGG net
class VGG(nn.Module):
    def __init__(self, in_channels=1, num_classes=10, 
                 input_size : int = 224, 
                 conv_arch : tuple = None) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.conv_arch = conv_arch
        
        self.conv, self.out_channels = self._vgg_block()
        
        if input_size == 224:
            self.size = 7
        elif input_size == 96:
            self.size = 3
        
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.out_channels * self.size * self.size, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, self.num_classes)
        )
        
# import torchvision.models as models
# models.vgg11()   
    def _vgg_block(self):
        layers = []
        for (num_convs, out_channels) in self.conv_arch:
            layers_in = []
            for _ in range(num_convs):
                layers_in.append(nn.Conv2d(self.in_channels, out_channels, kernel_size=3, padding=1))
                layers_in.append(nn.ReLU())
                self.in_channels = out_channels
            layers_in.append(nn.MaxPool2d(kernel_size=2, stride=2))
            layers.append(nn.Sequential(*layers_in))
        return nn.Sequential(*layers), out_channels
    
    def forward(self, x):
        x = self.conv(x)

        return self.linear(x)
    
    def print_layers(self):
        x = torch.randn(1, 1, 224, 224)
        for layer in self.conv:
            if layer.__class__.__name__ == 'Sequential':
                print('In Sequential:')
                for sub_layer in layer:
                    x = sub_layer(x)
                    print(sub_layer.__class__.__name__, 'output shape:\t', x.shape)
            # x = layer(x)
            # print(layer.__class__.__name__, 'output shape:\t', x.shape)
            print('-------------------')
        for layer in self.linear:
            x = layer(x)
            print(layer.__class__.__name__, 'output shape:\t', x.shape)
        
    

    

def vgg11(in_channels=1, num_classes=10, coef :int = 1, input_size=224):
    conv_arch = (1, 64//coef), (1, 128//coef), (2, 256//coef), (2, 512//coef), (2, 512//coef)
    return VGG(in_channels, num_classes, input_size, conv_arch)




def vgg13(in_channels=1, num_classes=10, input_size=224):
    conv_arch = ((2, 64), (2, 128), (2, 256), (2, 512), (2, 512))
    return VGG(in_channels, num_classes, input_size, conv_arch)

def vgg16(in_channels=1, num_classes=10, input_size=224):
    conv_arch = ((2, 64), (2, 128), (3, 256), (3, 512), (3, 512))
    return VGG(in_channels, num_classes, input_size, conv_arch)

def vgg19(in_channels=1, num_classes=10, input_size=224):
    conv_arch = ((2, 64), (2, 128), (4, 256), (4, 512), (4, 512))
    return VGG(in_channels, num_classes, input_size, conv_arch)

# net = vgg11(1, 10, input_size=96).cuda()
# x = torch.randn(1, 1, 96, 96).cuda()
# import torchinfo
# torchinfo.summary(net, (1, 1, 96, 96)) 
# y = net(x)

# NiN
class NiN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10) -> None:
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.net = nn.Sequential(
            self._nin_block(self.in_channels, 96, kernel_size=11, stride=4, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=2),
            self._nin_block(96, 256, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            self._nin_block(256, 384, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(p=0.5),
            self._nin_block(384, self.num_classes, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
    
    
    def _nin_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.net(x)
    

class NiN_Simple(nn.Module):
    def __init__(self, in_channels=1, num_classes=10) -> None:
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.net = nn.Sequential(
            self._nin_block(self.in_channels, 96, kernel_size=11, stride=4, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=2),
            self._nin_block(96, 256, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            self._nin_block(256, 384, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(p=0.5),
            self._nin_block(384, self.num_classes, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
    
    
    def _nin_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.ReLU(),
            # nn.Conv2d(out_channels, out_channels, kernel_size=1),
            # nn.ReLU()
        )
    
    def forward(self, x):
        return self.net(x)
    
    
# net = NiN_Simple(1, 10).cuda()
# import torchinfo
# torchinfo.summary(net, (1, 1, 224, 224))
# GoogleLeNet

# Inception block
class Inception(nn.Module):
    # c1--c4是每条路径的输出通道数
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # 线路1，单1x1卷积层
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # 线路2，1x1卷积层后接3x3卷积层
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路3，1x1卷积层后接5x5卷积层
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 线路4，3x3最大汇聚层后接1x1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        # 在通道维度上连结输出
        return torch.cat((p1, p2, p3, p4), dim=1)
    
    

class GoogLeNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        
        self.b1 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
                           nn.ReLU(),
                           nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        
        self.b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                           nn.ReLU(),
                           nn.Conv2d(64, 192, kernel_size=3, padding=1),
                           nn.ReLU(),
                           nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        
        self.b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                   Inception(256, 128, (128, 192), (32, 96), 64),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        
        self.b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                   Inception(512, 160, (112, 224), (24, 64), 64),
                   Inception(512, 128, (128, 256), (24, 64), 64),
                   Inception(512, 112, (144, 288), (32, 64), 64),
                   Inception(528, 256, (160, 320), (32, 128), 128),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                   Inception(832, 384, (192, 384), (48, 128), 128),
                   nn.AdaptiveAvgPool2d((1,1)),
                   nn.Flatten())
        self.linear = nn.Linear(1024, num_classes)
        
    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.linear(x)
        return x
    
# net = GoogleLeNet(1, 10).cuda()
# import torchinfo
# torchinfo.summary(net, (128, 1, 32, 32))
    
# models for 7.4.1
# question01: add batchnorm layer
class BNInception(nn.Module):
    # c1--c4是每条路径的输出通道数
    def __init__(self, in_channels, c1, c2, c3, c4):
        super().__init__()
        # 线路1，单1x1卷积层
        self.p1_1 = nn.Sequential(nn.Conv2d(in_channels, c1, kernel_size=1),
                      nn.BatchNorm2d(c1))
        self.p2_1 = nn.Sequential(nn.Conv2d(in_channels, c2[0], kernel_size=1),
                      nn.BatchNorm2d(c2[0]))
        self.p2_2 = nn.Sequential(nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1),
                      nn.BatchNorm2d(c2[1]))
        self.p3_1 = nn.Sequential(nn.Conv2d(in_channels, c3[0], kernel_size=1),
                      nn.BatchNorm2d(c3[0]))
        self.p3_2 = nn.Sequential(nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2),
                      nn.BatchNorm2d(c3[1]))
        self.p4_1 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                      nn.BatchNorm2d(in_channels))
        self.p4_2 = nn.Sequential(nn.Conv2d(in_channels, c4, kernel_size=1),
                      nn.BatchNorm2d(c4))

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        # 在通道维度上连结输出
        return torch.cat((p1, p2, p3, p4), dim=1)
    
class GoogLeNetBN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        
        self.b1 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
                           nn.BatchNorm2d(64),
                           nn.ReLU(),
                           nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        
        self.b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                           nn.BatchNorm2d(64),
                           nn.ReLU(),
                           nn.Conv2d(64, 192, kernel_size=3, padding=1),
                           nn.BatchNorm2d(192),
                           nn.ReLU(),
                           nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        
        self.b3 = nn.Sequential(BNInception(192, 64, (96, 128), (16, 32), 32),
                                BNInception(256, 128, (128, 192), (32, 96), 64),
                                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        
        self.b4 = nn.Sequential(BNInception(480, 192, (96, 208), (16, 48), 64),
                                BNInception(512, 160, (112, 224), (24, 64), 64),
                                BNInception(512, 128, (128, 256), (24, 64), 64),
                                BNInception(512, 112, (144, 288), (32, 64), 64),
                                BNInception(528, 256, (160, 320), (32, 128), 128),
                                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.b5 = nn.Sequential(BNInception(832, 256, (160, 320), (32, 128), 128),
                                BNInception(832, 384, (192, 384), (48, 128), 128),
                                nn.AdaptiveAvgPool2d((1,1)),
                                nn.Flatten())
        self.linear = nn.Linear(1024, num_classes)
        
    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.linear(x)
        return x

#question02: 对Incepetion进行调整

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
    
class Inception_v3_1(nn.Module):
    def __init__(self, in_channels, pool_channels, cut = False) -> None:
        super().__init__()
        self.cut = cut
        if not self.cut:
            self.b1 = BasicConv2d(in_channels, 64, kernel_size=1)
            
            self.b2 = nn.Sequential(BasicConv2d(in_channels, 48, kernel_size=1),
                                    BasicConv2d(48, 64, kernel_size=5, padding=2))
            
            self.b3 = nn.Sequential(BasicConv2d(in_channels, 64, kernel_size=1),
                                    BasicConv2d(64, 96, kernel_size=3, padding=1),
                                    BasicConv2d(96, 96, kernel_size=3, padding=1))
            
            self.b4 = BasicConv2d(in_channels, pool_channels, kernel_size=1)
        
        else:
            self.b1 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2)
            
            self.b2 = nn.Sequential(BasicConv2d(in_channels, 64, kernel_size=1),
                                    BasicConv2d(64, 96, kernel_size=3, padding=1),
                                    BasicConv2d(96, 96, kernel_size=3, stride=2))
        
    def forward(self, x):
        if not self.cut:
            b1 = self.b1(x)
            b2 = self.b2(x)
            b3 = self.b3(x)
            b4 = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
            b4 = self.b4(b4)
            return torch.cat((b1, b2, b3, b4), dim=1)
        else:
            b1 = self.b1(x)
            b2 = self.b2(x)
            b3 = F.max_pool2d(x, kernel_size=3, stride=2)
            return torch.cat((b1, b2, b3), dim=1)
        
        


class Inception_v3_2(nn.Module):
    def __init__(self, in_channel, channel_7x7, cut = False) -> None:
        super().__init__()
        
        self.cut = cut
        if not self.cut:
            c7 = channel_7x7
            
            self.b1 = BasicConv2d(in_channel, 192, kernel_size=1)
            
            self.b2 = nn.Sequential(BasicConv2d(in_channel, c7, kernel_size=1),
                                    BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3)),
                                    BasicConv2d(c7, 192, kernel_size=(7, 1), padding=(3, 0)))
            
            self.b3 = nn.Sequential(BasicConv2d(in_channel, c7, kernel_size=1),
                                    BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0)),
                                    BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3)),
                                    BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0)),
                                    BasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3)))
            
            self.b4 = BasicConv2d(in_channel, 192, kernel_size=1)
        
        else:
            self.b1 = nn.Sequential(BasicConv2d(in_channel, 192, kernel_size=1),
                                    BasicConv2d(192, 320, kernel_size=3, stride=2))
            
            self.b2 = nn.Sequential(BasicConv2d(in_channel, 192, kernel_size=1),
                                    BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3)),
                                    BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0)),
                                    BasicConv2d(192, 192, kernel_size=3, stride=2))
            
        
    def forward(self, x):
        if not self.cut:
            b1 = self.b1(x)
            b2 = self.b2(x)
            b3 = self.b3(x)
            b4 = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
            b4 = self.b4(b4)
            return torch.cat((b1, b2, b3, b4), dim=1)
        else:
            b1 = self.b1(x)
            b2 = self.b2(x)
            b3 = F.max_pool2d(x, kernel_size=3, stride=2)
            return torch.cat((b1, b2, b3), dim=1)
        
class Inception_v3_3(nn.Module):
    def __init__(self, in_channel) -> None:
        super().__init__()
        
        self.b1 = BasicConv2d(in_channel, 320, kernel_size=1)

        self.b2_1 = BasicConv2d(in_channel, 384, kernel_size=1)
        self.b2_2 = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.b2_3 = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))
        
        self.b3_1 = nn.Sequential(BasicConv2d(in_channel, 448, kernel_size=1),
                                BasicConv2d(448, 384, kernel_size=3, padding=1))
        self.b3_2 = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.b3_3 = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))
        
        self.b4 = BasicConv2d(in_channel, 192, kernel_size=1)
        
    def forward(self, x):
        
        b1 = self.b1(x)
        
        b2_in = self.b2_1(x)
        b2_mid = [self.b2_2(b2_in), self.b2_3(b2_in)]
        b2 = torch.cat(b2_mid, dim=1)
        
        b3_in = self.b3_1(x)
        b3_mid = [self.b3_2(b3_in), self.b3_3(b3_in)]
        b3 = torch.cat(b3_mid, dim=1)
        
        b4_in = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        b4 = self.b4(b4_in)
        
        return torch.cat((b1, b2, b3, b4), dim=1)
    
    
class Inception_V3_Net(nn.Module):
    def __init__(self, in_channel, num_classes = 10) -> None:
        super().__init__()
        
        self.conv1 = nn.Sequential(BasicConv2d(in_channel, 32, kernel_size=3, stride=2),
                                   BasicConv2d(32, 32, kernel_size=3),
                                   BasicConv2d(32, 64, kernel_size=3, padding=1),
                                   nn.MaxPool2d(kernel_size=3, stride=2))
                
        self.conv2 = nn.Sequential(BasicConv2d(64, 80, kernel_size=1),
                                      BasicConv2d(80, 192, kernel_size=3),
                                      nn.MaxPool2d(kernel_size=3, stride=2))
        
        # InceptionA
        self.incept1a = Inception_v3_1(192, 32)
        self.incept1b = Inception_v3_1(256, 64)
        self.incept1c = Inception_v3_1(288, 64)
        
        # InceptionB
        self.incept1_ = Inception_v3_1(288, None, cut=True)
        
        # InceptionC
        self.incept2a = Inception_v3_2(768, 128)
        self.incept2b = Inception_v3_2(768, 160)
        self.incept2c = Inception_v3_2(768, 160)
        self.incept2d = Inception_v3_2(768, 192)
        
        # InceptionD
        self.incept2_ = Inception_v3_2(768, None, cut=True)
        
        # InceptionE
        self.incept3a = Inception_v3_3(1280)
        self.incept3b = Inception_v3_3(2048)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.5)
        self.linear = nn.Linear(2048, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        
        x = self.incept1a(x)
        x = self.incept1b(x)
        x = self.incept1c(x)
        
        x = self.incept1_(x)
        x = self.incept2a(x)
        x = self.incept2b(x)
        x = self.incept2c(x)
        x = self.incept2d(x)
        
        x = self.incept2_(x)
        x = self.incept3a(x)
        x = self.incept3b(x)
        
        x = self.avgpool(x)
        x = self.dropout(x)
        x = x.view(-1, 2048)
        
        # print(x.shape)
        x = self.linear(x)
        return x
        

# import torch
# import torchinfo        
# mode = Inception_V3_Net(1, 10).cuda()
# torchinfo.summary(mode, (1, 1, 299, 299))
# y = mode(n)
        


# Resnet size = 32x32
import torch
from typing import Callable, List, Optional, Type, Union
import torch.nn.functional as F



def conv3x3(inplanes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(inplanes, 
                     out_planes, 
                     kernel_size=3, 
                     stride=stride, 
                     padding=1, 
                     bias=False)
# 构建Resnet模型，按照源代码，先构建BasicBlock，Bottleneck两个模块

class BasicBlock(nn.Module):
    expansion: int = 1  # 对输出通道进行倍增
    def __init__(self, 
                 inplanes: int,
                 planes: int,
                 stride: int = 1):
        super().__init__()
        
        
        # 此处3x3卷积层的操作遵循原始代码的操作
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        # 没有stride
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # prepare the identity for shortcut
        identity = x
        # 第一个卷积层
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        # 第二个卷积层
        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(identity)
        out = F.relu(out)
        
        return out
    

class Bottleneck(nn.Module):
    
    expansion: int = 4  # 对输出通道进行倍增
    
    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes) 
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        out += self.shortcut(identity)
        out = F.relu(out)
        return out
    

class ResNet(nn.Module):
    def __init__(self, 
                 in_channels,
                 block: Type[Union[BasicBlock, Bottleneck]],
                 layers: List[int],
                 num_classes: int = 100):
        super().__init__()
        
        self.inplanes = 64
        
        # 统一采样3x3卷积层，去掉这里的maxpooling
        self.conv1 = conv3x3(in_channels, 64)   # rgb, out = channels=64, stride=0
        self.bn1 = nn.BatchNorm2d(64)
        
        
        # 层次化设计
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)  # 对应着conv2_x
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # 对应着conv3_x
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # 对应着conv4_x
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # 对应着conv5_x
        
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        

        
    def _make_layer(self,
                    block: Type[Union[BasicBlock, Bottleneck]],
                    planes: int,  # 输入的通道
                    blocks: int, # 模块数目
                    stride: int = 1, # 步长
                    ) -> nn.Sequential:
        # 除了第一个stride由stride决定，其余都是1
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplanes, planes, stride))
            self.inplanes = planes * block.expansion
            
        # 传入layers的列表进入Sequential，并进行拆分成逐元素输入Sequential
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x= self.conv1(x)
        x = self.bn1(x)
        out = F.relu(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # 直接进行平均池化
        out = F.avg_pool2d(out, 4)
        # 拉成一维
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out    
    
def ResNet18(num_classes=10):
    return ResNet(3, BasicBlock, [2,2,2,2], num_classes)

def ResNet34(num_classes=10):
    return ResNet(3, BasicBlock, [3,4,6,3], num_classes)

def ResNet50(num_classes=10):
    return ResNet(3, Bottleneck, [3,4,6,3], num_classes)

def ResNet101(num_classes=10):
    return ResNet(3, Bottleneck, [3,4,23,3], num_classes)

def ResNet152(num_classes=10):
    return ResNet(3, Bottleneck, [3,8,36,3], num_classes)


# net = ResNet18(10).cuda()
# import torchinfo
# torchinfo.summary(net, (1, 3, 224, 224))


class DenseBlock(nn.Module):
    def __init__(self, in_channels, num_channels, num_convs) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_channels = num_channels
        self.num_convs = num_convs
        
        layers = []
        for i in range(self.num_convs):
            layers.append(self._conv_block(self.in_channels + i * self.num_channels, self.num_channels))
        
        self.net = nn.Sequential(*layers)
            
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
    
    def forward(self, x):
        for blk in self.net:
            y = blk(x)
            x = torch.cat((x, y), dim=1)
        return x


class Densenet(nn.Module):
    def __init__(self, in_channels, num_classes) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.b1 = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        num_channels, growth_rate = 64, 32
        num_convs_in_dense_blocks = [4, 4, 4, 4]
        blks = []
        for idx, num_conv in enumerate(num_convs_in_dense_blocks):
            blks.append(DenseBlock(num_channels, growth_rate, num_conv))
            
            num_channels += num_conv * growth_rate
            
            if idx != len(num_convs_in_dense_blocks) - 1:
                blks.append(self._transition_block(num_channels, num_channels // 2))
                num_channels = num_channels // 2
                
        self.b2 = nn.Sequential(*blks)
        
        self.b3 = nn.Sequential(nn.BatchNorm2d(num_channels), nn.ReLU(),
                                nn.AdaptiveAvgPool2d((1, 1)),
                                nn.Flatten(),
                                nn.Linear(num_channels, self.num_classes))

    def _transition_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
    
    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        return x
    
# net = Densenet(1, 10).cuda()
# import torchinfo
# torchinfo.summary(net, (1, 1, 224, 224))
        

# 稠密块
class DenseBlock(nn.Module):
    def __init__(self, in_channels, num_channels, num_convs) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_channels = num_channels
        self.num_convs = num_convs
        
        layers = []
        for i in range(self.num_convs):
            layers.append(self._conv_block(self.in_channels + i * self.num_channels, self.num_channels))
        self.net = nn.Sequential(*layers)

    # 卷积        
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
    
    # 前向传播
    def forward(self, x):
        for blk in self.net:
            y = blk(x)
            # 跨层连接通道维度上每个块的输入和输出
            x = torch.cat((x, y), dim=1)
        return x


class Densenet(nn.Module):
    def __init__(self, in_channels, num_classes) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        # DenseNet首先使用同ResNet一样的单卷积层和最大汇聚层
        self.b1 = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # num_channels为当前的通道数
        num_channels, growth_rate = 64, 32
        num_convs_in_dense_blocks = [4, 4, 4, 4]
        blks = []
        for idx, num_conv in enumerate(num_convs_in_dense_blocks):
            blks.append(DenseBlock(num_channels, growth_rate, num_conv))
            # 上一个稠密块的输出通道数
            num_channels += num_conv * growth_rate
            # 在稠密块之间添加一个转换层，使通道数量减半
            if idx != len(num_convs_in_dense_blocks) - 1:
                blks.append(self._transition_block(num_channels, num_channels // 2))
                num_channels = num_channels // 2
                
        self.b2 = nn.Sequential(*blks)
        
        self.b3 = nn.Sequential(nn.BatchNorm2d(num_channels), nn.ReLU(),
                                nn.AdaptiveAvgPool2d((1, 1)),
                                nn.Flatten(),
                                nn.Linear(num_channels, self.num_classes))

    # 过渡层
    def _transition_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
    
    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        return x
    
# net = Densenet(1, 10).cuda()
# import torchinfo
# torchinfo.summary(net, (1, 1, 224, 224))

# 练习7.7.4
from collections import OrderedDict

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        # 将输入输出两通道拼接
        return torch.cat([x, new_features], 1)

# 稠密块
class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)

# 过渡层
class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    def __init__(self, in_channels = 3,growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):

        super(DenseNet, self).__init__()
        self.in_channels = in_channels
        # 第一个卷积层
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(self.in_channels, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # 四层 依次经过稠密块和过渡层
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # 最后一个bn层
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # 线性层
        self.classifier = nn.Linear(num_features, num_classes)

        # 网络初始化设置
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        out = self.classifier(out)
        return out

def densenet121(**kwargs):
    model = DenseNet(3, num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16), **kwargs)
    return model


def densenet169(**kwargs):
    model = DenseNet(3, num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32), **kwargs)
    return model


def densenet201(**kwargs):
    model = DenseNet(3, num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32), **kwargs)
    return model


def densenet161(**kwargs):
    model = DenseNet(3, num_init_features=96, growth_rate=48, block_config=(6, 12, 64, 48), **kwargs)
    return model


net = densenet121(num_classes=10).cuda()
import torchinfo
torchinfo.summary(net, (1, 3, 224, 224))
