from __future__ import division
from math import sqrt as sqrt
from itertools import product as product
import torch

# 最终, 输出的ouput就是一张图片中所有的default box的坐标, 对于论文中的默认设置来说产生的box数量为:
# 382×4+192×6+102×6+52×6+32×4+12×4=8732
class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """
    def __init__(self, cfg):
        # 在SSD的init中, cfg=(coco, voc)[num_classes=3]
        # coco, voc的相关配置都来自于data/cfg.py 文件
        super(PriorBox, self).__init__()       
        self.image_size = cfg['min_dim']
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        self.version = cfg['name']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    # 至于aspect ratio，用ar表示为下式：注意这里一共有5种aspect ratio
    # ar = {1, 2, 3, 1/2, 1/3}
    # 因此每个default box的宽的计算公式为： wk = sk * sqrt(ar)
    # 高的计算公式为：hk = sk / sqrt(ar)（很容易理解宽和高的乘积是scale的平方）
    # 另外当aspect ratio为1时，作者还增加一种scale的default box： sk = sqrt(sk * s(k+1))
    # 因此，对于每个feature map cell而言，一共有6种或4种default box。
    # 可以看出这种default box在不同的feature层有不同的scale，在同一个feature层又有不同的aspect ratio，
    # 因此基本上可以覆盖输入图像中的各种形状和大小的object！
    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps): # 存放的是feature map的尺寸:38,19,10,5,3,1
            for i, j in product(range(f), repeat=2): # 这里实际上可以用最普通的for循环嵌套来代替, 主要目的是产生anchor的坐标(i,j)
                f_k = self.image_size / self.steps[k] # steps=[8,16,32,64,100,300]. f_k大约为feature map的尺寸
                # unit center x,y
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k # 这里一定要特别注意 i,j 和cx, cy的对应关系, 因为cy对应的是行, 所以应该零cy与i对应.

                # aspect_ratios 为1时对应的box
                # 'min_sizes': [30, 60, 111, 162, 213, 264], 
                s_k = self.min_sizes[k]/self.image_size
                mean += [cx, cy, s_k, s_k]

                # 根据原文, 当 aspect_ratios 为1时, 会有一个额外的 box, 如下:
                # rel size: sqrt(s_k * s_(k+1))
                # 'max_sizes': [60, 111, 162, 213, 264, 315], 
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # 其余(2, 或 2,3)的宽高比(aspect ratio)
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output # 输出default box坐标(可以理解为anchor box)
