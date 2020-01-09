import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import voc, coco
import os

# SSD网络是由 VGG 网络后接 multibox 卷积层 组成的, 每一个 multibox 层会有如下分支:
    # - 用于class conf scores的卷积层
    # - 用于localization predictions的卷积层
    # - 与priorbox layer相关联, 产生默认的bounding box

    # 参数:
    # phase: test/train
    # size: 输入图片的尺寸
    # base: VGG16的层
    # extras: 将输出结果送到multibox loc和conf layers的额外的层
    # head: "multibox head", 包含一系列的loc和conf卷积层.
class SSD(nn.Module):  # 自定义SSD网络
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, head, num_classes):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        # {'num_classes': 21, 
        # 'lr_steps': (80000, 100000, 120000), 
        # 'max_iter': 120000, 
        # 'feature_maps': [38, 19, 10, 5, 3, 1],
        #  'min_dim': 300, 
        # 'steps': [8, 16, 32, 64, 100, 300], 
        # 'min_sizes': [30, 60, 111, 162, 213, 264], 
        # 'max_sizes': [60, 111, 162, 213, 264, 315], 
        # 'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]], 
        # 'variance': [0.1, 0.2],
        #  'clip': True,
        #  'name': 'VOC'}
        self.cfg = (coco, voc)[num_classes == 3]
        self.priorbox = PriorBox(self.cfg) # layers/functions/prior_box.py class PriorBox(object)
        with torch.no_grad():
            self.priors = Variable(self.priorbox.forward())
        self.size = size

        # SSD network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20) # layers/modules/l2norm.py class L2Norm(nn.Module)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0]) # head = (loc_layers, conf_layers)
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            # 用于将预测结果转换成对应的坐标和类别编号形式, 方便可视化.
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45) #  layers/functions/detection.py class Detect

    # 参数: x, 输入的batch 图片, Shape: [batch, 3, 300, 300]

        # 返回值: 取决于不同阶段
        # test: 预测的类别标签, confidence score, 以及相关的location.
        #       Shape: [batch, topk, 7]
        # train: 关于以下输出的元素组成的列表
        #       1: confidence layers, Shape: [batch*num_priors, num_classes]
        #       2: localization layers, Shape: [batch, num_priors*4]
        #       3: priorbox layers, Shape: [2, num_priors*4]
    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        # sources是不同层用来预测的特征图
        sources = list()
        # loc用来预测坐标位置回归
        loc = list()
        # conf用来预测置信度
        conf = list()

        # 计算vgg直到conv4_3的relu
        for k in range(23):
            x = self.vgg[k](x)

        s = self.L2Norm(x)
        # 将 conv4_3 的特征层输出添加到 sources 中, 后面会根据 sources 中的元素进行预测
        sources.append(s)
      
        # 将vgg应用到fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        # 计算extras layers, 并且将结果存储到sources列表中
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            # 在extras_layers中, 第1,3,5,7,9(从第0开始)的卷积层的输出会用于预测box位置和类别, 因此, 将其添加到 sources列表中
            if k % 2 == 1:
                sources.append(x)

        # 应用multibox到source layers上, source layers中的元素均为各个用于预测的特征图谱
        # zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
        # 所以分别将source，loc, con对应的元素，组成元祖，将元祖组成列表
        # 注意pytorch中卷积层的输入输出维度是:[N×C×H×W]
        for (x, l, c) in zip(sources, self.loc, self.conf):
            # permute重新排列维度顺序, PyTorch维度的默认排列顺序为 (N, C, H, W),
            # 因此, 这里的排列是将其改为 $(N, H, W, C)$.
            # contiguous返回内存连续的tensor, 由于在执行permute或者transpose等操作之后, tensor的内存地址可能不是连续的,
            # 然后 view 操作是基于连续地址的, 因此, 需要调用contiguous语句.
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        # cat 是 concatenate 的缩写, view返回一个新的tensor, 具有相同的数据但是不同的size, 类似于numpy的reshape
        # 在调用view之前, 需要先调用contiguous
        # 将除batch以外的其他维度合并, 因此, 对于边框坐标来说, 最终的shape为(两维):[batch, num_boxes*4]
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        # 同理, 最终的shape为(两维):[batch, num_boxes*num_classes]
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
    
        if self.phase == "test":
            # 这里用到了 detect 对象, 该对象主要用于接预测出来的结果进行解析, 以获得方便可视化的边框坐标和类别编号
            output = self.detect(
                loc.view(loc.size(0), -1, 4), #  又将shape转换成: [batch, num_boxes, 4], 即[1, 8732, 4]
                self.softmax(conf.view(conf.size(0), -1, # 同理,  shape 为[batch, num_boxes, num_classes], 即 [1, 8732, 3]
                             self.num_classes)),               
                self.priors.type(type(x.data)) # 利用 PriorBox对象获取特征图谱上的 default box, 该参数的shape为: [8732,4].
            )
        else:
            output = ( # 如果是训练阶段, 则无需解析预测结果, 直接返回然后求损失.
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file): # 加载权重文件
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False): # 搭建VGG网络
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    # 池化层pool5将原来的stride=2的2*2变成了stride=1的3*3
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    # conv6采用扩展卷积或带孔卷积，其在不增加参数和模型复杂度的条件下指数级扩大卷积的视野，其使用扩张率(dilation rate)参数，来表示扩张的大小
    # conv6采用3*3大小，dilation_rate = 6的扩张卷积
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers

# exts1_1 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1)
# exts1_2 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)
# exts2_1 = nn.Conv2d(512, 128, 1, 1, 0)
# exts2_2 = nn.Conv2d(128, 256, 3, 2, 1)
# exts3_1 = nn.Conv2d(256, 128, 1, 1, 0)
# exts3_2 = nn.Conv2d(128, 256, 3, 1, 0)
# exts4_1 = nn.Conv2d(256, 128, 1, 1, 0)
# exts4_2 = nn.Conv2d(128, 256, 3, 1, 0)
def add_extras(cfg, i, batch_norm=False): # 向VGG网络中添加额外的层
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S': # （1， 3)[True] = 3, (1, 3)[False] = 1
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers


def multibox(vgg, extra_layers, cfg, num_classes): # 构建multibox结构
    # cfg = [4, 6, 6, 6, 4, 4]
    # num_classes = 3
    # ssd总共会选择6个卷积特征图谱进行预测, 分别为, vggnet的conv4_3, conv7 以及extras_layers的4段卷积的输出(每段由两个卷积层组成, 具体可看extras_layers的实现).
    # 也就是说, loc_layers 和 conf_layers 分别具有6个预测层.
    loc_layers = []
    loc_layers = []
    conf_layers = []
    vgg_source = [21, -2] # 21 denote conv4_3, -2 denote conv7
    for k, v in enumerate(vgg_source):
        # 定义6个坐标预测层, 输出的通道数就是每个像素点上会产生的 default box 的数量
        # loc1 = nn.Conv2d(vgg[21].out_channels, 4*4, 3, 1, 1) # 利用conv4_3的特征图谱, 也就是 vgg 网络 List 中的第 21 个元素的输出(注意不是第21层, 因为这中间还包含了不带参数的池化层).
        # loc2 = nn.Conv2d(vgg[-2].out_channels, 6*4, 3, 1, 1) # Conv7
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        # 定义分类层, 和定位层差不多, 只不过输出的通道数不一样, 因为对于每一个像素点上的每一个default box,
        # 都需要预测出属于任意一个类的概率, 因此通道数为 default box 的数量乘以类别数. 
        # conf1 = nn.Conv2d(vgg[21].out_channels, 4*num_classes, 3, 1, 1)
        # conf2 = nn.Conv2d(vgg[-2].out_channels, 6*num_classes, 3, 1, 1)
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):
        # loc3 = nn.Conv2d(vgg[1].out_channels, 6*4, 3, 1, 1) # exts1_2
        # loc4 = nn.Conv2d(extras[3].out_channels, 6*4, 3, 1, 1) # exts2_2
        # loc5 = nn.Conv2d(extras[5].out_channels, 4*4, 3, 1, 1) # exts3_2
        # loc6 = nn.Conv2d(extras[7].out_channels, 4*4, 3, 1, 1) # exts4_2
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        # conf3 = nn.Conv2d(extras[1].out_channels, 6*num_classes, 3, 1, 1)
        # conf4 = nn.Conv2d(extras[3].out_channels, 6*num_classes, 3, 1, 1)
        # conf5 = nn.Conv2d(extras[5].out_channels, 4*num_classes, 3, 1, 1)
        # conf6 = nn.Conv2d(extras[7].out_channels, 4*num_classes, 3, 1, 1)
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    # loc_layers = [loc1, loc2, loc3, loc4, loc5, loc6]
    # conf_layers = [conf1, conf2, conf3, conf4, conf5, conf6]
    return vgg, extra_layers, (loc_layers, conf_layers)


base = { # VGG网络结构参数
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
}
extras = { # extras层参数
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}
mbox = { # multibox相关参数
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [],
}


def build_ssd(phase, size=300, num_classes=3): # 构建模型函数，调用上面的函数进行构建
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 300: # 仅支持300size的SSD
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 (size=300) is supported!")
        return
    base_, extras_, head_ = multibox(vgg(base[str(size)], 3),
                                     add_extras(extras[str(size)], 1024),
                                     mbox[str(size)], num_classes)
    return SSD(phase, size, base_, extras_, head_, num_classes)
