from torch import nn
import torch
from torch.autograd import Variable
import math
from net_setting import current_setting as NET_SETTING

import math
from torch.nn import Parameter
import torch.nn.functional as F

class HardSigmoid(nn.Module):
    r""" HardSigmoid = ReLU6(input + 3)/6
    """
    # def __init__(self):
    #     super(HardSigmoid,self).__init__()

    def forward(self, input):
        x = F.relu6(input + 3)/6
        # print("HardSigmoid.return :   {}".format(x))
        return x

class HardSwish(nn.Module):
    r""" HardSwish = input * HardSigmoid(input)
    """
    # def __init__(self):
    #     super(HardSwish,self).__init__()

    def forward(self, input):
        hswish = HardSigmoid()
        x = input * hswish(input)
        # print("HardSwish.return :   {}".format(x))
        return x

class ArcMarginProduct(nn.Module):
    def __init__(self, in_features=128, out_features=200, s=32.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        # init.kaiming_uniform_()
        # self.weight.data.normal_(std=0.001)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


_AVAILABLE_ACTIVATION = {
            "relu": nn.ReLU(),
            "relu6": nn.ReLU6(),
            "hswish": HardSwish(),
            "hsigmoid": HardSigmoid(),
            "softmax": nn.Softmax(),
            "prelu": nn.PReLU(),
        }


def _get_layer(layer_name, layer_dict, param1=0):
    if layer_name is None:
        raise NotImplementedError(f"Layer_name [{layer_name}] is None !")
    if layer_name in layer_dict.keys():
        if layer_name == "prelu":
            layer = nn.PReLU(param1)
        else:
            layer = layer_dict.get(layer_name)
        return layer
    else:
        raise NotImplementedError(f"Layer [{layer_name}] is not implemented")

class ConvNormAct(nn.Module):
    r"""
    * WFaceNet main Conv-Norm-Act
    * param@ inp : input_channels : Number of channels in the input image
    * param@ oup : out_channels (int): Number of channels produced by the convolution
    * param@ k : kernel_size (int default=3): Size of the convolving kernel
    * param@ s : stride (int default=1): Stride of the convolution.
    * param@ p : padding (int defaylt=0): Zero-padding added to both sides of the input.
    * param@ dw : Whether to use Depthwise Conv.
    * param@ act_layer : Which use activation function( one of the _AVAILABLE_ACTIVATION )
    """
    def __init__(self, 
                inp:int, 
                oup:int, 
                kernel_size=3, 
                stride:int=1, 
                padding:int=0, 
                norm_layer:str="None", # is not "None" , will use nn.BatchNorm2d 
                act_layer:str="None", # get target nonlinear_function from _AVAILABLE_ACTIVATION dict 
                use_bias:bool=True, 
                l2_reg:float=1e-5, 
                dw:bool=False,
                name:str="ConvNormAct",
    ):
        super(ConvNormAct, self).__init__()

        self.norm = (norm_layer != "None")  # True:        use batchNormalization
        self.linear = (act_layer == "None") # True: do not use activation layer.

        if dw:
            self.conv = nn.Conv2d(inp,oup,kernel_size,stride,padding,groups=inp,bias=use_bias)
        else:
            self.conv = nn.Conv2d(inp,oup,kernel_size,stride,padding,bias=use_bias)

        if self.norm:   
            self.bn = nn.BatchNorm2d(oup)

        if not self.linear:
            self.act = _get_layer(act_layer,_AVAILABLE_ACTIVATION,oup)


    def forward(self,x):
        x = self.conv(x)
        if self.norm:
            x = self.bn(x)
        if self.linear:
            return x
        else:
            x =self.act(x)
            return x

class SEBottleneck(nn.Module):
    def __init__(self,inp,oup,reduction:int=4,l2_reg=0.01,name="SEBottleneck",):
        super(SEBottleneck,self).__init__()
        self.reduction = reduction
        self.l2_reg = l2_reg
        pooling_output_size = 1
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.squeeze = ConvNormAct(inp=inp, oup=inp//self.reduction, kernel_size=1, stride=1, padding=0, norm_layer="None", act_layer="relu", use_bias=False, l2_reg=l2_reg, name="Squeeze")
        self.excite = ConvNormAct(inp=inp//self.reduction,oup=oup,kernel_size=1,norm_layer="None", act_layer="hsigmoid", use_bias=False, l2_reg=self.l2_reg,name="Excite")
    
    def forward(self,x):
        inputs = x
        x = self.gap(x)
        x = self.squeeze(x)
        x = self.excite(x)
        return inputs * x

class ScoreLayer(nn.Module):
    def __init__(self,inp,oup,reduction:int=4,l2_reg=0.01,name="ScoreLayer",):
        super(ScoreLayer,self).__init__()
        self.reduction = reduction
        self.l2_reg = l2_reg
        pooling_output_size = 1
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.squeeze = ConvNormAct(inp=inp, oup=inp//self.reduction, kernel_size=1, stride=1, padding=0, norm_layer="None", act_layer="relu", use_bias=False, l2_reg=l2_reg, name="Squeeze")
        self.excite = ConvNormAct(inp=inp//self.reduction,oup=oup,kernel_size=1,norm_layer="None", act_layer="hsigmoid", use_bias=False, l2_reg=self.l2_reg,name="Excite")
    
    def forward(self,x):
        x = self.gap(x)
        x = self.squeeze(x)
        x = self.excite(x)
        return x

class Bottleneck(nn.Module):
    def __init__(self,
                inp, 
                oup, 
                exp, 
                k, 
                stride, 
                use_se, 
                act_layer="None", 
                l2_reg:float=1e-5,
    ):
        super(Bottleneck,self).__init__()

        self.inp = inp
        self.oup = oup
        self.stride = stride
        self.use_se = use_se
        self.act_layer = act_layer

        # Expand                    i,  o,k,s,p
        self.expand = ConvNormAct(inp,exp,1,1,0,"bn",act_layer,False,l2_reg)

        # Depthwise
        dw_padding = (k - 1)//2
        self.depthwise_norm = ConvNormAct(inp=exp,oup=exp,kernel_size=k,stride=stride,padding=dw_padding,norm_layer="bn",use_bias=False,dw=True)

        if self.use_se:
            self.se = SEBottleneck(exp,exp,l2_reg=l2_reg,name="dw/SEBottleneck")

        self.act = _get_layer(self.act_layer,_AVAILABLE_ACTIVATION,exp)

        # Project
        self.project = ConvNormAct(exp,oup,kernel_size=1,norm_layer="bn",act_layer="None",use_bias=False,l2_reg=l2_reg,name="Project")

    def forward(self,x):
        inputs = x
        x = self.expand(x)
        x = self.depthwise_norm(x)
        if self.use_se:
            x = self.se(x)
        x = self.act(x)
        x = self.project(x)

        if self.stride == 1 and self.inp == self.oup:
            return inputs + x
        else:
            return x

class WLayer(nn.Module):
    def __init__(self,inp,oup,name="WLayer"):
        super(WLayer,self).__init__()
        self.dwconv = ConvNormAct(inp=inp,oup=oup,kernel_size=3,stride=1,padding=1,norm_layer="bn",act_layer="prelu",use_bias=False,dw=True)
        self.sl = SEBottleneck(inp,oup)

    def forward(self,x):
        x1 = self.dwconv(x)
        x2 = self.sl(x)
        x = x1 * x2
        return x

        
class WFaceNet(nn.Module):
    def __init__(self,bottleneck_setting=NET_SETTING, name="WFaceNet."):
        super(WFaceNet, self).__init__()
        self.name = name
        self.bneck_setting = bottleneck_setting
        first_expand_to_channels = 16

        self.first = ConvNormAct(inp=3,oup=first_expand_to_channels,kernel_size=3,stride=1,padding=1,norm_layer="bn",act_layer="prelu",use_bias=False,name="FirstLayer")

        # self.wlayer = WLayer(inp=first_expand_to_channels,oup=first_expand_to_channels) # 等价于下面两行
        self.firstdw = ConvNormAct(inp=first_expand_to_channels,oup=first_expand_to_channels,kernel_size=3,stride=1,padding=1,norm_layer="bn",act_layer="prelu",use_bias=False,dw=True)
        self.se = SEBottleneck(first_expand_to_channels,first_expand_to_channels)

        self.inplanes = first_expand_to_channels
        self.bnecks = self._make_layer(Bottleneck,self.bneck_setting)
        last_bneck_oup = self.bneck_setting[-1][0]

        self.conv2 = ConvNormAct(last_bneck_oup,512,1,1,0,norm_layer="bn",act_layer="prelu",use_bias=False)
        self.dwlinear7 = ConvNormAct(512,512,(7,7),1,0,dw=True,norm_layer="bn",use_bias=False)
        self.linear1 = ConvNormAct(512,128,1,1,0,norm_layer="bn",use_bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.au = 'ClannXy'

    def _make_layer(self, block, setting):
        layers = []

        for oup, exp, k, s, SE, NL in setting:
            layers.append(block(self.inplanes,oup,exp,k,s,SE,NL))
            self.inplanes=oup

        return nn.Sequential(*layers)

    def forward(self,x):
        x = self.first(x)
        x = self.firstdw(x)*self.se(x) # x = self.wlayer(x)
        x = self.bnecks(x)
        x = self.conv2(x)
        x = self.dwlinear7(x)
        x = self.linear1(x)
        x = x.view(x.size(0), -1)

        return x

def display_flops_params(net):
    input = Variable(torch.FloatTensor(2, 3, 112, 112))
    from thop import profile
    flops, params = profile(net, inputs=(input, ))
    print('flops: %sm'%(flops/1000000))
    print('params: %sm'%(params/1000000))

def display_params(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('paramerters: {}.'.format(num_params))



if __name__ == "__main__":
    # inputs = Variable(torch.FloatTensor(2, 3, 112, 112))
    one = torch.FloatTensor(2,3,112,112)
    net = WFaceNet()
    embedding = net(one)
    # print(embedding)
    display_params(net)
    # display_flops_params(net)
    pass