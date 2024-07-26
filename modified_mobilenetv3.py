import torch
import torch.nn as nn
from collections import OrderedDict


class SKModel(nn.Module):
    def __init__(self, channel=512, kernels=[3, 5], reduction=2, group=1, L=32):
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
        self.cardinality=2
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
     
        self.fc = nn.Sequential(
            nn.Linear(in_channel//self.reduction*self.cardinality, in_channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y1 = self.fc1(y)
        y2 = self.fc2(y)
        y_concate = torch.cat([y1,y2],dim=1)
        y_ex_dim = self.fc(y_concate).view(b,c,1,1)

        return x * y_ex_dim.expand_as(x)


def conv_block_mo(in_channel, out_channel, kernel_size=3, strid=1, groups=1,
               activation="h-swish"):  
    padding = (kernel_size - 1) // 2  
    assert activation in ["h-swish", "relu"]  
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size, strid, padding=padding, groups=groups, bias=False),  # conv
        nn.BatchNorm2d(out_channel),  # bn
        nn.Hardswish(inplace=True) if activation == "h-swish" else nn.ReLU(inplace=True)  # h-swish/relu
    )



class SEblock(nn.Module): 
    def __init__(self, channel):  
        super(SEblock, self).__init__() 

        self.channel = channel 
        self.attention = nn.Sequential(  
            nn.AdaptiveAvgPool2d(1),  
            nn.Conv2d(self.channel, self.channel // 4, 1, 1, 0),  
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel // 4, self.channel, 1, 1, 0),  
            nn.Hardswish(inplace=True) 
        )

    def forward(self, x):
        a = self.attention(x) 
        return a * x


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


class bneck(nn.Module): 
    def __init__(self, in_channel, out_channel, kernel_size=3, strid=1, t=6., se=True, activation="h-swish", ks=False, ca=False):  # 初始化方法
        super(bneck, self).__init__() 
        self.in_channel = in_channel  
        self.out_channel = out_channel  
        self.kernel_size = kernel_size  
        self.strid = strid  
        self.t = t  
        self.hidden_channel = int(in_channel * t) 
        self.se = se  
        self.activation = activation 

        layers = []
        if self.t != 1:  
            layers += [conv_block_mo(self.in_channel, self.hidden_channel, kernel_size=1,
                                  activation=self.activation)] 
        layers += [conv_block_mo(self.hidden_channel, self.hidden_channel, kernel_size=self.kernel_size, strid=self.strid,
                              groups=self.hidden_channel,
                              activation=self.activation)]  
        if self.se:  
            layers += [SEblock(self.hidden_channel)]  
        layers += [conv_block_mo(self.hidden_channel, self.out_channel, kernel_size=1)[:-1]]  
        self.residul_block = nn.Sequential(*layers)  
        self.sk = sk
        self.ca = ca
        if self.sk:
            self.sk_model = skModel(out_channel)
        if self.ca:
            self.ca_model = CoordAtt(out_channel, out_channel)

    def forward(self, x): 
        if self.strid == 1 and self.in_channel == self.out_channel: 
            out = x + self.residul_block(x)  # x+F(x)
        else:
            out = self.residul_block(x)  # F(x)
        if self.sk:
            out = self.sk_model(out) + out
        if self.ca:
            out = self.ca_model(out) + out
        return out


class MobileNetV3(nn.Module):
    def __init__(self, num_classes, model_size="large", ks=False, ca=False, tr=False):  
        super(MobileNetV3, self).__init__() 

        self.num_classes = num_classes  
        self.tr = tr
        assert model_size in ["small", "large"]  
        self.model_size = model_size
        if self.model_size == "small":  
            self.feature = nn.Sequential(  # 特征提取部分
                conv_block_mo(3, 16, strid=2, activation="h-swish"),  # conv+bn+h-swish,(n,3,224,224)-->(n,16,112,112)
                bneck(16, 16, kernel_size=3, strid=2, t=1, se=True, activation="relu"),
                # bneck,(n,16,112,112)-->(n,16,56,56)
                bneck(16, 24, kernel_size=3, strid=2, t=4.5, se=False, activation="relu"),
                # bneck,(n,16,56,56)-->(n,24,28,28)
                bneck(24, 24, kernel_size=3, strid=1, t=88 / 24, se=False, activation="relu", sk=sk),
                # bneck,(n,24,28,28)-->(n,24,28,28)
                bneck(24, 40, kernel_size=5, strid=2, t=4, se=True, activation="h-swish"),
                # bneck,(n,24,28,28)-->(n,40,14,14)
                bneck(40, 40, kernel_size=5, strid=1, t=6, se=True, activation="h-swish"),
                # bneck,(n,40,14,14)-->(n,40,14,14)
                bneck(40, 40, kernel_size=5, strid=1, t=6, se=True, activation="h-swish"),
                # bneck,(n,40,14,14)-->(n,40,14,14)
                bneck(40, 48, kernel_size=5, strid=1, t=3, se=True, activation="h-swish"),
                # bneck,(n,40,14,14)-->(n,48,14,14)
                bneck(48, 48, kernel_size=5, strid=1, t=3, se=True, activation="h-swish", sk=sk),
                # bneck,(n,48,14,14)-->(n,48,14,14)
                bneck(48, 96, kernel_size=5, strid=2, t=6, se=True, activation="h-swish"),
                # bneck,(n,48,14,14)-->(n,96,7,7)
                bneck(96, 96, kernel_size=5, strid=1, t=6, se=True, activation="h-swish"),
                # bneck,(n,96,7,7)-->(n,96,7,7)
                bneck(96, 96, kernel_size=5, strid=1, t=6, se=True, activation="h-swish", ca=ca),
                # bneck,(n,96,7,7)-->(n,96,7,7)
                conv_block_mo(96, 576, kernel_size=1, activation="h-swish")
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
                    bneck(24, 24, kernel_size=3, strid=1, t=3, se=False, activation="relu", sk=sk),
                    # bneck,(n,24,56,56)-->(n,24,56,56)
                )
                self.feature2 = nn.Sequential(
                    bneck(24, 40, kernel_size=5, strid=2, t=3, se=True, activation="relu"),
                    # bneck,(n,24,56,56)-->(n,40,28,28)
                    bneck(40, 40, kernel_size=5, strid=1, t=3, se=True, activation="relu"),
                    # bneck,(n,40,28,28)-->(n,40,28,28)
                    bneck(40, 40, kernel_size=5, strid=1, t=3, se=True, activation="relu", sk=sk),
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
                    bneck(24, 24, kernel_size=3, strid=1, t=3, se=False, activation="relu", sk=sk),
                    # bneck,(n,24,56,56)-->(n,24,56,56)
                    bneck(24, 40, kernel_size=5, strid=2, t=3, se=True, activation="relu"),
                    # bneck,(n,24,56,56)-->(n,40,28,28)
                    bneck(40, 40, kernel_size=5, strid=1, t=3, se=True, activation="relu"),
                    # bneck,(n,40,28,28)-->(n,40,28,28)
                    bneck(40, 40, kernel_size=5, strid=1, t=3, se=True, activation="relu", sk=sk),
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

    def forward(self, x):  
        if self.tr:
            x1 = self.feature1(x)  
            x2 = self.feature2(x1)  
            x3 = self.feature3(x2)  
            x4 = self.feature4(x3) 
            x = self.tr_model(x1, x2, x3, x4)
        else:
            x = self.feature(x)  # 提取特征
        x = self.classifier(x)  # 分类
        return x.view(-1, self.num_classes)  # 压缩不需要的维度，返回分类结果,(n,num_classes,1,1)-->(n,num_classes)



