##  MobileNet是轻量级的神经网络，MobileNet v3 2019年发布
##############   MobileNet v1, v2   #########
    ##  深度可分离卷积： 将3x3卷积核分为3x1和1x3，这样将计算量(Mult-Adds)降低了1/8左右，参数(parameters)也更少
    ##  采用ReLU6激活函数，输入大于6则输出6，作者认为在低精度计算下具有更强的鲁棒性
    ##  低维度（通道数）使用ReLU更容易造成信息丢失，所以将最后的那个ReLU6换成Linear
    ##  深度卷积没有改变通道数的能力，所以在深度卷积前采用1x1卷积来扩充维度，深度卷积后再利用1x1降低维度
    ##  采用了shortcut结构
################    MobileNet v3    ############
    ##  将ReLU替换为Swish，仅这样就能提高了0.6-0.9个百分点。作者使用了h-Swish，用ReLU6来近似，减小计算量

##  MNIST数据集第一轮训练准确率就达到 98.1%
import torchvision
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F 
import numpy as np
import torch.utils.data as Data
from PIL import Image
import torchvision.transforms as transforms



##  3x3的卷积
def conv_bn(inp, oup, stride, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, kernel_size=3, stride=stride, padding=1, bias=False),
        norm_layer(oup), 
        nlin_layer(inplace=True)
    )

##  1x1的卷积
def conv_1x1_bn(inp, oup, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, kernel_size=1, stride=1, padding=0, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )

class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace
        self.ReLu6 = nn.ReLU6(inplace=self.inplace)
    def forward(self, x):
        return x * self.ReLu6(x+3) / 6.  ## 用这个来代替swish(x*Sigmoid(βx))

##  ReLu6输出范围是[0, 6]，所以除6进行归一化。x+3也是为了更好的模拟sigmoid，使得x=0使输出为0.5
##  Hsigmoid来代替Sigmoid，提高运行速度
class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace
        self.ReLu6 = nn.ReLU6(inplace=self.inplace)
    def forward(self, x):
        return self.ReLu6(x+3) / 6.  

##  注意SEModule结构，通过Avg将wxhxc -> 1x1xc.再经过两层linear。将这部分作为shortcut和原输入x相乘
class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            Hsigmoid()
        )
    def forward(self, x):
        b, c, w, h = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x) ########################
    
class Identity(nn.Module):
    def __init__(self, channel):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

##  这个函数的作用是将x转换成8的整数（不足往上增）
def make_divisible(x, divisible_by=8):
    return int(np.ceil(x * 1. / divisible_by) * divisible_by) ## np.ceil将float转换为整数，余数向上取整

class MobileBottleneck(nn.Module):
    def __init__(self, inp, oup, kernel, stride, exp, se=False, nl='RE'):
        super(MobileBottleneck, self).__init__()
        assert stride in [1, 2]
        assert kernel in [3, 5]
        padding = (kernel - 1) // 2
        self.use_res_connect = stride == 1 and inp == oup  ## 输入输出通道数和wh不变，则采用shortcut
        
        conv_layer = nn.Conv2d
        norm_layer = nn.BatchNorm2d
        if nl == 'RE':
            nlin_layer = nn.ReLU  ##  or ReLU6
        elif nl == 'HS':
            nlin_layer = Hswish
        else:
            raise NotImplementedError
        if se:
            SELayer = SEModule
        else:
            SELayer = Identity
        ##  1x1 -> 3x3 -> 1x1
        ##  exp是通过第一次1x1卷积增加至的通道数
        self.conv = nn.Sequential(
            ##  pw
            conv_layer(inp, exp, kernel_size=1, stride=1, padding=0, bias=False),
            norm_layer(exp),
            nlin_layer(inplace=True),
            ##  dw
            conv_layer(exp, exp, kernel_size=kernel, stride=stride, padding=padding, groups=exp, bias=True),
            norm_layer(exp),
            SELayer(exp),
            nlin_layer(exp), 
            ##  pw_linear
            conv_layer(exp, oup, kernel_size=1, stride=1, padding=0, bias=False),
            norm_layer(oup),
        )
    def forward(self, x):
        if self.use_res_connect:  ##  shortcut
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV3(nn.Module):
    def __init__(self, n_class=10, input_size=224, dropout=0.8, mode='small', width_mult=1.0):
        super(MobileNetV3, self).__init__()
        input_channel = 16
        last_channel = 1280
        ##  k是kernel_size, exp是输出的通道数， c是out_channel, nl是激活函数
        ##  s是stride, se = True是SEModule, False是Identity
        if mode == 'large':
            mobile_setting = [
            ##    k, exp,  c,   se,    nl,  s,
                [ 3, 16,  16,  False, 'RE', 1],
                [ 3, 64,  24,  False, 'RE', 2],
                [ 3, 72,  24,  False, 'RE', 1],
                [ 5, 72,  40,  True,  'RE', 2],
                [ 5, 120, 40,  True,  'RE', 1],
                [ 5, 120, 40,  True,  'RE', 1],
                [ 3, 240, 80,  False, 'HS', 2],
                [ 3, 200, 80,  False, 'HS', 1],
                [ 3, 184, 80,  False, 'HS', 1],
                [ 3, 184, 80,  False, 'HS', 1],
                [ 3, 480, 112, True,  'HS', 1],
                [ 3, 672, 112, True,  'HS', 1],
                [ 5, 672, 160, True,  'HS', 2],
                [ 5, 960, 160, True,  'HS', 1],
                [ 5, 960, 160, True,  'HS', 1]
            ]
        elif mode == 'small':
            mobile_setting = [
            ##    k, exp, c,    se,    nl,  s,
                [ 3, 16,  16,  True,  'RE', 2],
                [ 3, 72,  24,  False, 'RE', 2],
                [ 3, 88,  24,  False, 'RE', 1],
                [ 5, 96,  40,  True,  'HS', 2],
                [ 5, 240, 40,  True,  'HS', 1],
                [ 5, 240, 40,  True,  'HS', 1],
                [ 5, 120, 48,  True,  'HS', 1],
                [ 5, 144, 48,  True,  'HS', 1],
                [ 5, 288, 96,  True,  'HS', 2],
                [ 5, 576, 96,  True,  'HS', 1],
                [ 5, 576, 96,  True,  'HS', 1]
            ]
        else:
            raise NotImplementedError

        ##  building first layer
        assert input_size % 32 == 0
        last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(1, input_channel, stride=2, nlin_layer=Hswish)] ## 这里用的是灰度图，所以是1
        self.classifier = []

        ##  building mobile blocks
        for k, exp, c, se , nl, s in mobile_setting:
            out_channel = make_divisible(c * width_mult)
            exp_channel = make_divisible(exp * width_mult)
            self.features.append(MobileBottleneck(input_channel, out_channel, k ,s, exp_channel, se, nl))
            input_channel = out_channel
        
        ##  最后两层将通道数 960 -> 960 x width_mult -> 1280  (最后加上层Linear -> 1000)
        if mode == 'large':
            last_conv = make_divisible(960 * width_mult)
            self.features.append(conv_1x1_bn(input_channel, last_conv, nlin_layer=Hswish))
            self.features.append(nn.AdaptiveAvgPool2d(1)) ## w,h为1,1
            self.features.append(nn.Conv2d(last_conv, last_channel, 1, 1, 0))
            self.features.append(Hswish(inplace=True))

        elif mode == 'small':
            last_conv = make_divisible(576 * width_mult)
            self.features.append(conv_1x1_bn(input_channel, last_conv, nlin_layer=Hswish))
            self.features.append(nn.AdaptiveAvgPool2d(1))
            self.features.append(nn.Conv2d(last_conv, last_channel, 1, 1, 0))
            self.features.append(Hswish(inplace=True))

        else:
            raise NotImplementedError

        self.features = nn.Sequential(*self.features)

        ##  classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(last_channel, n_class)
        ) 

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)    ## 用mean将w,h -> 1,1(avgpool也可以，但)
        x = self.classifier(x)
        return x


def mobilenetv3(predtrain=False, **kwargs):
    model = MobileNetV3(**kwargs)
    if predtrain: ##  找不到资源
        state_dict = torch.load('mobilenetv3_small')
        model.load_state_dict(state_dict, strict=True)
    return model


model = mobilenetv3().cuda()
## 数据处理
transform = transforms.Compose([
    transforms.Resize(224), ## 224表示将图片最小的边resize到224，另一个边按相同的比例缩放
    transforms.ToTensor()
])

## parameters
LR = 0.0001
Batch_size = 32
EPOCH = 1

## data_loader
train_data = torchvision.datasets.MNIST(
    root='../mnist',  ##表示MNIST数据集下载/已存在的位置，../表示是相对于当前py文件上一级目录的mnist文件夹
    train=True,
    transform=transform,
    download=False  ## 如果没有下载就改为True自动下载
)
train_loader = Data.DataLoader(dataset=train_data, batch_size=Batch_size, shuffle=True)
test_data = torchvision.datasets.MNIST(root='../mnist', train=False)
test_y = test_data.test_labels[:2000] ## volatile=True表示依赖这个节点的所有节点都不会进行反向求导，用于测试集
test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1), volatile=True).type(torch.FloatTensor)[:2000]
test_x_1 = torch.empty(test_x.size(0), 1, 224, 224)
for i,v in enumerate(test_x):   
    temp = v[0].numpy()
    temp = Image.fromarray(temp)
    #temp = transforms.ToPILImage()(v[0])  ## 自己动手将tensor转换为Image，不要用这个函数，血的教训
    temp = transforms.Resize((224,224))(temp)
    temp = np.array(temp)
    temp = torch.Tensor(temp)/255.
    test_x_1[i][0] = temp
test_x = test_x_1.cuda()
test_x_1 = 0
test_x.volatile = True

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

## train
for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        x = Variable(x).cuda()
        y = Variable(y).cuda()
        out_put = model(x)
        optimizer.zero_grad()
        loss = loss_func(out_put, y)
        loss.backward()
        optimizer.step()
        if step%100 == 0:
            print(step)
 
model.eval() 
accuracy = 0
for i, v in enumerate(test_x): ##  直接将2000张图片扔进去内存会不足
    test = v.unsqueeze(0)
    test_output = model(test)
    pred_y = torch.max(test_output, 1)[1].cpu().data.squeeze()
    accuracy += 1 if pred_y == test_y[i] else 0
accuracy /= len(test_y)
print('Epoch:', epoch, '|train loss:%.4f' % loss.item(), '|test accuracy:',accuracy)
