# -*- coding: utf-8 -*-
import torch

# 将 boxes 的坐标信息转换成左上角和右下角的形式
def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    # 将(cx, cy, w, h) 形式的box坐标转换成 (xmin, ymin, xmax, ymax) 形式
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                     boxes[:, :2] + boxes[:, 2:]/2), 1)  # xmax, ymax


def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, 2:] + boxes[:, :2])/2,  # cx, cy
                     boxes[:, 2:] - boxes[:, :2], 1)  # w, h

# 返回 box_a 与 box_b 集合中元素的交集
def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    # box_a: (truths), (tensor:[num_obj, 4])
    # box_b: (priors), (tensor:[num_priors, 4], 即[8732, 4])
    # return: (tensor:[num_obj, num_priors]) box_a 与 box_b 两个集合中任意两个 box 的交集, 其中res[i][j]代表box_a中第i个box与box_b中第j个box的交集.(非对称矩阵)
    # 思路: 先将两个box的维度扩展至相同维度: [num_obj, num_priors, 4], 然后计算面积的交集
    # 两个box的交集可以看成是一个新的box, 该box的左上角坐标是box_a和box_b左上角坐标的较大值, 右下角坐标是box_a和box_b的右下角坐标的较小值
    A = box_a.size(0)
    B = box_b.size(0)
    # a中某个box与b中某个box的左上角的较大者
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    # 求右下角的较小者
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    # 右下角减去左上角，如果是负值，说明没有交集，置为0
    inter = torch.clamp((max_xy - min_xy), min=0)
    # 高* 宽，返回交集的面积
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    #  A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    # box_a: (truths),(tensor:[num_obj, 4])
    # box_b: (priors), (tensor:[num_priors, 4])
    # return : (tensor(num_obj, num_priors)),  代表了box_a 和box_b两个集合中任意两个box之间的交并比
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B] 返回任意两个box之间的交并比，res[i][j]代表box_a中的第i个box与box_b中的第j个box之间的交并比


def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index

0        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    # threshold: (float)确定是否匹配的交并比阀值
    # truths： （tensor:[num_obj, 4]存储真实box的边框坐标
    # priors: (tensor: [num_priors, 4], 即[8732, 4]）,存储推荐框的坐标，注意，此时的框是 default box，而不是SSD网络预测出来的框的坐标
    # variances： cfg['variance'],[0.1, 0.2]
    # labels: (tensor: [num_obj]),代表每个真实box对应的类别编号
    # loc_t: (tensor: [batch_size, 8732, 4])
    # conf_t: (tensor: [batch_size, 8732])
    # idx: batch_size中图片的序号
    # jaccard index
    # 返回任意两个box之间的交并比，overlaps[i][j] 代表box_a中的第i个box与box_b中的第j个box之间的交并比.
    overlaps = jaccard(
        truths,
        point_form(priors)
    )
    # (Bipartite Matching)
    # [1,num_objects] best prior for each ground truth
    # 二部图匹配
    # [num_objs, 1]，得到对于每个gt box来说的匹配度最高的prior box,前者存储交并比，后者存储prior box在num_prior中的位置
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    # [1,num_priors] best ground truth for each prior
    # 【1, num_priors]，即[1, 8732]，同理，得到对于每个prior box来说的匹配度最高的gt box
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    # 将输入张量形状中的1 去除并返回。
    # best_truth_idx维度压缩后变为[num_priors]
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    # best_prior_idx维度为[num_objs]
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)

    # 该语句会将与gt box 匹配度最好的prior box 的交并比置为2，确保其最大，以免防止某些gtbox没有匹配的priorbox
    # 按参数index中的索引数确定的顺序，将原tensor用参数val值填充。
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior
    # TODO refactor: index  best_prior_idx with long tensor
    # ensure every gt matches with its prior of max overlap
    # 假想一种极端情况，所有的priorbox与某个gtbox（标记为G）的交并比为1，而其他gtbox分别有一个
    # 交并比最高的priorbox，但是肯定小于1（因为其他的gtbox与G的交并比肯定小于1），这样一来，就会
    # 使得所有的priorbox都与G匹配，为了防止这种情况，我们将那些对gtbox来说，具有最高交并比的priorbox，
    # 强制进行互相匹配，即令best_truth_idx[best_prior_idx[j]] = j
    # 注意：因为gtbox的数量远远少于priorbox的数量，因此，同一个gtbox会与多个priorbox匹配
    for j in range(best_prior_idx.size(0)): # range: 0~num_obj - 1
        best_truth_idx[best_prior_idx[j]] = j
        # best_prior_idx[j]代表与box_a的第j个box交并比最高的priorbox的下标，将与该gtbox
        # 匹配度最高的prior box的下标改为j，由此，完成了该gtbox与第j个prior box的匹配
        # 这里的循环只会进行num_obj次，剩余的匹配为best_truth_idx 中原本的值
        # 这里处理的情况是，prior box中第i个box与gtbox中第k个box的交并比最高
        # 即 best_truth_idx[i] = k
        # 但是对于best_prior_idx[k]来说，它却与priorbox的第l个box有着最高的交并比
        # 即best_prior_idx[k]=l
        # 而对于gtbox的另一个边框gtbox[j]来说，它与priorbox[i]的交并比最大
        # 但是对于best_prior_idx[j] = i
        # 那么，此时，我们就应该将best_truth_idx[i]=k修改称best_truth_idx[i]=j
        # 即令priorbox[i]与gtbox[j]对应
        # 这样做的原因：防止某个gtbox没有匹配的prior box
    matches = truths[best_truth_idx]          # Shape: [num_priors,4]
    # truths的shape为[num_objs, 4]，而best_truth_idx是一个指示下标的列表，列表长度为8732
    # 列表中的下标范围为0~num_objs-1,代表的是与每个priorbox匹配的gtbox的下标
    # 上面的表达式会返回一个shape为[num_priors,4],即[8372,4]的tensor，代表就是与每个priorbox匹配的gtbox的坐标值
    # 这里得到的是每个priorbox匹配的类别编号，shape为[8732]
    conf = labels[best_truth_idx] + 1         # Shape: [num_priors]
    # 将与gtbox的交并比小于阀值的置为0，即认为是非物体框
    conf[best_truth_overlap < threshold] = 0  # label as background
    # 返回编码后的中心坐标和宽高
    loc = encode(matches, priors, variances)
    # 设置第idx张图片的gt编码坐标信息
    loc_t[idx] = loc    # [num_priors,4] encoded offsets to learn
    # 设置第idx张图片的编号信息（大于0即为物体编号，认为有物体，小于0认为是背景)
    conf_t[idx] = conf  # [num_priors] top class label for each prior


def encode(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 4].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """
    # 对边框坐标进行编码，需要宽度方差和高度方差两个参数
    # matched: [num_priors, 4]存储的是与priorbox匹配的gtbox的坐标，形式为（xmin, ymin, xmax, ymax)
    # priors: [num_priors, 4]存储的是priorbox的坐标，形式为（cx, cy, w, h)
    # return: encoded boxes;[num_priors, 4]
    # dist b/t match center and prior's center
    # 用互相匹配的gtbox的中心坐标减去priorbox的中心坐标，获取中心坐标的偏移量
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]
    # encode variance
    # 令中心坐标分别除以 d_i^w 和 d_i^h, 正如原文公式所示
    # variances[0]为0.1, 令其分别乘以w和h, 得到d_i^w 和 d_i^h
    g_cxcy /= (variances[0] * priors[:, 2:])
    # match wh / prior wh
    # 令互相匹配的gtbox的宽高除以priorbox的宽高
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    # variances[1] = 0.2
    g_wh = torch.log(g_wh) / variances[1]
    # return target for smooth_l1_loss
    # 将编码后的中心坐标和宽高连接起来，返回[num_priors, 4]
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]


# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """
    # loc_data: [num_priors, 4], [8732, 4]
    # prior_data: [num_priors, 4], [8732, 4] 目标预选框
    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max


# Original author: Francisco Massa:
# https://github.com/fmassa/object-detection.torch
# Ported to PyTorch by Max deGroot (02/01/2017)
def nms(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """

    keep = scores.new(scores.size(0)).zero_().long()
    # numel函数返回数组中元素的个数
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # 升序排序
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        # 沿着指定维度对输入进行切片，取index中指定的相应项，然后返回到一个新的张量
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # 计算交并比
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count
