import torch
import torch.nn as nn
from collections import OrderedDict


class KSModel(nn.Module):
    def __init__(self, channel=512, kernels=[1, 3, 5, 7], reduction=2, group=1, L=32):
        super().__init__()
        self.d = max(L, channel // reduction)
        self.convs = nn.ModuleList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(channel, channel, kernel_size=k, padding=k // 2, groups=group)),
                    ('bn', nn.BatchNorm2d(channel)),
                    ('relu', nn.ReLU())
                ]))
            )
        self.fc = nn.Linear(channel, self.d)
        self.fcs = nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d, channel))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        bs, c, _, _ = x.size()
        conv_outs = []
        ### split
        for conv in self.convs:
            conv_outs.append(conv(x))
        feats = torch.stack(conv_outs, 0)  # k,bs,channel,h,w

        ### fuse
        U = sum(conv_outs)  # bs,c,h,w

        ### reduction channel
        S = U.mean(-1).mean(-1)  # bs,c
        Z = self.fc(S)  # bs,d

        ### calculate attention weight
        weights = []
        for fc in self.fcs:
            weight = fc(Z)
            weights.append(weight.view(bs, c, 1, 1))  # bs,channel
        attention_weughts = torch.stack(weights, 0)  # k,bs,channel,1,1
        attention_weughts = self.softmax(attention_weughts)  # k,bs,channel,1,1

        ### fuse
        V = (attention_weughts * feats).sum(0)
        return V


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=2):
        super(CoordAtt, self).__init__()
        # self.sae = SaELayer(inp)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # x = self.sae(x)
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return a_w * a_h

class SaELayer(nn.Module):
    def __init__(self, in_channel, reduction=4):
        super(SaELayer, self).__init__()
        assert in_channel>=reduction and in_channel%reduction==0,'invalid in_channel in SaElayer'
        self.reduction = reduction
        self.cardinality=4
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        #cardinality 1
        self.fc1 = nn.Sequential(
            nn.Linear(in_channel,in_channel//self.reduction, bias=False),
            nn.ReLU(inplace=True)
        )
        # cardinality 2
        self.fc2 = nn.Sequential(
            nn.Linear(in_channel, in_channel // self.reduction, bias=False),
            nn.ReLU(inplace=True)
        )
        # cardinality 3
        self.fc3 = nn.Sequential(
            nn.Linear(in_channel, in_channel // self.reduction, bias=False),
            nn.ReLU(inplace=True)
        )
        # cardinality 4
        self.fc4 = nn.Sequential(
            nn.Linear(in_channel, in_channel // self.reduction, bias=False),
            nn.ReLU(inplace=True)
        )

        self.fc = nn.Sequential(
            nn.Linear(in_channel//self.reduction*self.cardinality, in_channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y1 = self.fc1(y)
        y2 = self.fc2(y)
        y3 = self.fc3(y)
        y4 = self.fc4(y)
        y_concate = torch.cat([y1,y2,y3,y4],dim=1)
        y_ex_dim = self.fc(y_concate).view(b,c,1,1)

        return x * y_ex_dim.expand_as(x)


def conv_block_mo(in_channel, out_channel, kernel_size=3, strid=1, groups=1,
               activation="h-swish"):  # 定义卷积块,conv+bn+h-swish/relu
    padding = (kernel_size - 1) // 2  # 计算padding
    assert activation in ["h-swish", "relu"]  # 激活函数在h-swish和relu中选择
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size, strid, padding=padding, groups=groups, bias=False),  # conv
        nn.BatchNorm2d(out_channel),  # bn
        nn.Hardswish(inplace=True) if activation == "h-swish" else nn.ReLU(inplace=True)  # h-swish/relu
    )



class SEblock(nn.Module):  # 定义Squeeze and Excite注意力机制模块
    def __init__(self, channel):  # 初始化方法
        super(SEblock, self).__init__()  # 继承初始化方法

        self.channel = channel  # 通道数
        self.attention = nn.Sequential(  # 定义注意力模块
            nn.AdaptiveAvgPool2d(1),  # avgpool
            nn.Conv2d(self.channel, self.channel // 4, 1, 1, 0),  # 1x1conv，代替全连接
            nn.ReLU(inplace=True),  # relu
            nn.Conv2d(self.channel // 4, self.channel, 1, 1, 0),  # 1x1conv，代替全连接
            nn.Hardswish(inplace=True)  # h-swish，此处原文图中为hard-alpha，未注明具体激活函数，这里使用h-swish
        )

    def forward(self, x):  # 前传函数
        a = self.attention(x)  # 通道注意力权重
        return a * x  # 返回乘积


class HireAtt(nn.Module):
    def __init__(self, in_channels=960, out_channels=512, reduction=16):
        super(HireAtt, self).__init__()

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels, out_channels // reduction, 1, 1, 0)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels // reduction, out_channels, 1, 1, 0)
        self.sigmoid2 = nn.Sigmoid()

    def forward(self, x1, x2, x3, x4):
        gap1 = self.gap(x1)
        gap2 = self.gap(x2)
        gap3 = self.gap(x3)
        gap4 = self.gap(x4)
        gap = torch.concat([gap1, gap2, gap3, gap4], dim=1)
        x_out = self.conv1(gap)
        x_out = self.relu1(x_out)
        x_out = self.conv2(x_out)
        x_out = self.sigmoid2(x_out)
        x_out = x_out * x4
        return x_out


class bneck(nn.Module):  # 定义改进的倒置残差结构,对应原文中的bneck
    def __init__(self, in_channel, out_channel, kernel_size=3, strid=1, t=6., se=True, activation="h-swish", ks=False, ca=False):  # 初始化方法
        super(bneck, self).__init__()  # 继承初始化方法
        self.in_channel = in_channel  # 输入通道数
        self.out_channel = out_channel  # 输出通道数
        self.kernel_size = kernel_size  # 卷积核尺寸
        self.strid = strid  # 步长
        self.t = t  # 中间层通道扩大倍数，对应原文expansion ratio
        self.hidden_channel = int(in_channel * t)  # 计算中间层通道数
        self.se = se  # 是否使用SE注意力机制模块
        self.activation = activation  # 激活函数形式

        layers = []  # 存放模型结构
        if self.t != 1:  # 如果expansion ratio不为1
            layers += [conv_block_mo(self.in_channel, self.hidden_channel, kernel_size=1,
                                  activation=self.activation)]  # 添加conv+bn+h-swish/relu
        layers += [conv_block_mo(self.hidden_channel, self.hidden_channel, kernel_size=self.kernel_size, strid=self.strid,
                              groups=self.hidden_channel,
                              activation=self.activation)]  # 添加conv+bn+h-swish/relu，此处使用组数等于输入通道数的分组卷积实现depthwise conv
        if self.se:  # 如果使用SE注意力机制模块
            layers += [SEblock(self.hidden_channel)]  # 添加SEblock
        layers += [conv_block_mo(self.hidden_channel, self.out_channel, kernel_size=1)[:-1]]  # 添加1x1conv+bn，此处不再进行激活函数
        self.residul_block = nn.Sequential(*layers)  # 倒置残差结构块
        self.ks = ks
        self.ca = ca
        if self.ks:
            self.ks_model = KSModel(out_channel)
        if self.ca:
            self.ca_model = CoordAtt(out_channel, out_channel)

    def forward(self, x):  # 前传函数
        if self.strid == 1 and self.in_channel == self.out_channel:  # 如果卷积步长为1且前后通道数一致，则连接残差边
            out = x + self.residul_block(x)  # x+F(x)
        else:  # 否则不进行残差连接
            out = self.residul_block(x)  # F(x)
        if self.ks:
            out = self.ks_model(out) + out
        if self.ca:
            out = self.ca_model(out) + out
        return out


class MobileNetV3(nn.Module):  # 定义MobileNet v3网络
    def __init__(self, num_classes, model_size="large", ks=False, ca=False, tr=False):  # 初始化方法
        super(MobileNetV3, self).__init__()  # 继承初始化方法

        self.num_classes = num_classes  # 类别数量
        self.tr = tr
        assert model_size in ["small", "large"]  # 模型尺寸，仅支持small和large两种
        self.model_size = model_size  # 模型尺寸选择
        if self.model_size == "small":  # 如果是small模型
            self.feature = nn.Sequential(  # 特征提取部分
                conv_block_mo(3, 16, strid=2, activation="h-swish"),  # conv+bn+h-swish,(n,3,224,224)-->(n,16,112,112)
                bneck(16, 16, kernel_size=3, strid=2, t=1, se=True, activation="relu"),
                # bneck,(n,16,112,112)-->(n,16,56,56)
                bneck(16, 24, kernel_size=3, strid=2, t=4.5, se=False, activation="relu"),
                # bneck,(n,16,56,56)-->(n,24,28,28)
                bneck(24, 24, kernel_size=3, strid=1, t=88 / 24, se=False, activation="relu", ks=ks),
                # bneck,(n,24,28,28)-->(n,24,28,28)
                bneck(24, 40, kernel_size=5, strid=2, t=4, se=True, activation="h-swish"),
                # bneck,(n,24,28,28)-->(n,40,14,14)
                bneck(40, 40, kernel_size=5, strid=1, t=6, se=True, activation="h-swish"),
                # bneck,(n,40,14,14)-->(n,40,14,14)
                bneck(40, 40, kernel_size=5, strid=1, t=6, se=True, activation="h-swish"),
                # bneck,(n,40,14,14)-->(n,40,14,14)
                bneck(40, 48, kernel_size=5, strid=1, t=3, se=True, activation="h-swish"),
                # bneck,(n,40,14,14)-->(n,48,14,14)
                bneck(48, 48, kernel_size=5, strid=1, t=3, se=True, activation="h-swish", ks=ks),
                # bneck,(n,48,14,14)-->(n,48,14,14)
                bneck(48, 96, kernel_size=5, strid=2, t=6, se=True, activation="h-swish"),
                # bneck,(n,48,14,14)-->(n,96,7,7)
                bneck(96, 96, kernel_size=5, strid=1, t=6, se=True, activation="h-swish"),
                # bneck,(n,96,7,7)-->(n,96,7,7)
                bneck(96, 96, kernel_size=5, strid=1, t=6, se=True, activation="h-swish", ca=ca),
                # bneck,(n,96,7,7)-->(n,96,7,7)
                conv_block_mo(96, 576, kernel_size=1, activation="h-swish")
                # conv+bn+h-swish,(n,96,7,7)-->(n,576,7,7),此处没有使用SE注意力模块
            )

            self.classifier = nn.Sequential(  # 分类部分
                nn.AdaptiveAvgPool2d(1),  # avgpool,(n,576,7,7)-->(n,576,1,1)
                nn.Conv2d(576, 1024, 1, 1, 0),  # 1x1conv,(n,576,1,1)-->(n,1024,1,1)
                nn.Hardswish(inplace=True),  # h-swish
                nn.Conv2d(1024, self.num_classes, 1, 1, 0)  # 1x1conv,(n,1024,1,1)-->(n,num_classes,1,1)
            )
        else:
            if self.tr:
                self.feature1 = nn.Sequential(  # 特征提取部分
                    conv_block_mo(3, 16, strid=2, activation="h-swish"),
                    # conv+bn+h-swish,(n,3,224,224)-->(n,16,112,112)
                    bneck(16, 16, kernel_size=3, strid=1, t=1, se=False, activation="relu"),
                    # bneck,(n,16,112,112)-->(n,16,112,112)
                    bneck(16, 24, kernel_size=3, strid=2, t=4, se=False, activation="relu"),
                    # bneck,(n,16,112,112)-->(n,24,56,56)
                    bneck(24, 24, kernel_size=3, strid=1, t=3, se=False, activation="relu", ks=ks),
                    # bneck,(n,24,56,56)-->(n,24,56,56)
                )
                self.feature2 = nn.Sequential(
                    bneck(24, 40, kernel_size=5, strid=2, t=3, se=True, activation="relu"),
                    # bneck,(n,24,56,56)-->(n,40,28,28)
                    bneck(40, 40, kernel_size=5, strid=1, t=3, se=True, activation="relu"),
                    # bneck,(n,40,28,28)-->(n,40,28,28)
                    bneck(40, 40, kernel_size=5, strid=1, t=3, se=True, activation="relu", ks=ks),
                    # bneck,(n,40,28,28)-->(n,40,28,28)
                )
                self.feature3 = nn.Sequential(
                    bneck(40, 80, kernel_size=3, strid=2, t=6, se=False, activation="h-swish"),
                    # bneck,(n,40,28,28)-->(n,80,14,14)
                    bneck(80, 80, kernel_size=3, strid=1, t=2.5, se=False, activation="h-swish"),
                    # bneck,(n,80,14,14)-->(n,80,14,14)
                    bneck(80, 80, kernel_size=3, strid=1, t=2.3, se=False, activation="h-swish"),
                    # bneck,(n,80,14,14)-->(n,80,14,14)
                    bneck(80, 80, kernel_size=3, strid=1, t=2.3, se=False, activation="h-swish", ca=ca),
                    # bneck,(n,80,14,14)-->(n,80,14,14)
                    bneck(80, 112, kernel_size=3, strid=1, t=6, se=True, activation="h-swish"),
                    # bneck,(n,80,14,14)-->(n,112,14,14)
                    bneck(112, 112, kernel_size=3, strid=1, t=6, se=True, activation="h-swish"),
                    # bneck,(n,112,14,14)-->(n,112,14,14)
                    bneck(112, 160, kernel_size=5, strid=2, t=6, se=True, activation="h-swish", ca=ca),
                    # bneck,(n,112,14,14)-->(n,160,7,7)
                )
                self.feature4 = nn.Sequential(
                    bneck(160, 160, kernel_size=5, strid=1, t=6, se=True, activation="h-swish"),
                    # bneck,(n,160,7,7)-->(n,160,7,7)
                    bneck(160, 160, kernel_size=5, strid=1, t=6, se=True, activation="h-swish"),
                    # bneck,(n,160,7,7)-->(n,160,7,7)
                    conv_block_mo(160, 960, kernel_size=1, activation="h-swish")
                    # conv+bn+h-swish,(n,160,7,7)-->(n,960,7,7)
                )

            else:
                self.feature = nn.Sequential(  # 特征提取部分
                    conv_block_mo(3, 16, strid=2, activation="h-swish"),  # conv+bn+h-swish,(n,3,224,224)-->(n,16,112,112)
                    bneck(16, 16, kernel_size=3, strid=1, t=1, se=False, activation="relu"),
                    # bneck,(n,16,112,112)-->(n,16,112,112)
                    bneck(16, 24, kernel_size=3, strid=2, t=4, se=False, activation="relu"),
                    # bneck,(n,16,112,112)-->(n,24,56,56)
                    bneck(24, 24, kernel_size=3, strid=1, t=3, se=False, activation="relu", ks=ks),
                    # bneck,(n,24,56,56)-->(n,24,56,56)
                    bneck(24, 40, kernel_size=5, strid=2, t=3, se=True, activation="relu"),
                    # bneck,(n,24,56,56)-->(n,40,28,28)
                    bneck(40, 40, kernel_size=5, strid=1, t=3, se=True, activation="relu"),
                    # bneck,(n,40,28,28)-->(n,40,28,28)
                    bneck(40, 40, kernel_size=5, strid=1, t=3, se=True, activation="relu", ks=ks),
                    # bneck,(n,40,28,28)-->(n,40,28,28)
                    bneck(40, 80, kernel_size=3, strid=2, t=6, se=False, activation="h-swish"),
                    # bneck,(n,40,28,28)-->(n,80,14,14)
                    bneck(80, 80, kernel_size=3, strid=1, t=2.5, se=False, activation="h-swish"),
                    # bneck,(n,80,14,14)-->(n,80,14,14)
                    bneck(80, 80, kernel_size=3, strid=1, t=2.3, se=False, activation="h-swish"),
                    # bneck,(n,80,14,14)-->(n,80,14,14)
                    bneck(80, 80, kernel_size=3, strid=1, t=2.3, se=False, activation="h-swish", ca=ca),
                    # bneck,(n,80,14,14)-->(n,80,14,14)
                    bneck(80, 112, kernel_size=3, strid=1, t=6, se=True, activation="h-swish"),
                    # bneck,(n,80,14,14)-->(n,112,14,14)
                    bneck(112, 112, kernel_size=3, strid=1, t=6, se=True, activation="h-swish"),
                    # bneck,(n,112,14,14)-->(n,112,14,14)
                    bneck(112, 160, kernel_size=5, strid=2, t=6, se=True, activation="h-swish", ca=ca),
                    # bneck,(n,112,14,14)-->(n,160,7,7)
                    bneck(160, 160, kernel_size=5, strid=1, t=6, se=True, activation="h-swish"),
                    # bneck,(n,160,7,7)-->(n,160,7,7)
                    bneck(160, 160, kernel_size=5, strid=1, t=6, se=True, activation="h-swish"),
                    # bneck,(n,160,7,7)-->(n,160,7,7)
                    conv_block_mo(160, 960, kernel_size=1, activation="h-swish")  # conv+bn+h-swish,(n,160,7,7)-->(n,960,7,7)
                )

            if self.tr:
                self.tr_model = HireAtt(1184, 960)

            self.classifier = nn.Sequential(  # 分类部分
                nn.AdaptiveAvgPool2d(1),  # avgpool,(n,960,7,7)-->(n,960,1,1)
                nn.Conv2d(960, 1280, 1, 1, 0),  # 1x1conv,(n,960,1,1)-->(n,1280,1,1)
                nn.Hardswish(inplace=True),  # h-swish
                nn.Conv2d(1280, self.num_classes, 1, 1, 0)  # 1x1conv,(n,1280,1,1)-->(n,num_classes,1,1)
            )

    def forward(self, x):  # 前传函数
        if self.tr:
            x1 = self.feature1(x)  # 提取特征
            x2 = self.feature2(x1)  # 提取特征
            x3 = self.feature3(x2)  # 提取特征
            x4 = self.feature4(x3)  # 提取特征
            x = self.tr_model(x1, x2, x3, x4)
        else:
            x = self.feature(x)  # 提取特征
        x = self.classifier(x)  # 分类
        return x.view(-1, self.num_classes)  # 压缩不需要的维度，返回分类结果,(n,num_classes,1,1)-->(n,num_classes)



class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, kernel_size=k, stride=s, padding=0, groups=g, dilation=d, bias=False)
        self.bn = SyncBatchNorm(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            # 如果TransformerBlock，即ViT模块输入和输出通道不同，提前通过一个卷积层让通道相同
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)