# Dataset
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F
import torch.nn.functional as nnF

import numpy as np
import cv2
import random

from .mDataset import ImageCreatorDataset, ImageTeadDataset
from .utils import processGray

def correct_dims(*images):
    corr_images = []
    # print(images)
    for img in images:
        if len(img.shape) == 2:
            corr_images.append(np.expand_dims(img, axis=2))
        else:
            corr_images.append(img)

    if len(corr_images) == 1:
        return corr_images[0]
    else:
        return corr_images

def to_long_tensor(pic):
    # handle numpy array
    img = torch.from_numpy(np.array(pic, np.uint8))
    # backward compatibility
    return img.long()

class JointTransform2D:
    """
    Performs augmentation on image and mask when called. Due to the randomness of augmentation transforms,
    it is not enough to simply apply the same Transform from torchvision on the image and mask separetely.
    Doing this will result in messing up the ground truth mask. To circumvent this problem, this class can
    be used, which will take care of the problems above.

    Args:
        crop: tuple describing the size of the random crop. If bool(crop) evaluates to False, no crop will
            be taken.
        p_flip: float, the probability of performing a random horizontal flip.
        color_jitter_params: tuple describing the parameters of torchvision.transforms.ColorJitter.
            If bool(color_jitter_params) evaluates to false, no color jitter transformation will be used.
        p_random_affine: float, the probability of performing a random affine transform using
            torchvision.transforms.RandomAffine.
        long_mask: bool, if True, returns the mask as LongTensor in label-encoded format.
    """
    def __init__(self, crop=(32, 32), p_flip=0.5, color_jitter_params=(0.1, 0.1, 0.1, 0.1),
                 p_random_affine=0, long_mask=False, normTransform=None):
        self.crop = crop
        self.p_flip = p_flip
        self.color_jitter_params = color_jitter_params
        if color_jitter_params:
            self.color_tf = T.ColorJitter(*color_jitter_params)
        self.p_random_affine = p_random_affine
        self.long_mask = long_mask

        normMean, normStd = normTransform
        self.normTransform = T.Normalize(normMean, normStd)

    def __call__(self, image, mask):
        # transforming to PIL image
        image, mask = F.to_pil_image(image), F.to_pil_image(mask)

        # if F._get_image_size(image) == (256, 256):
        #     a = 1

        # random crop
        if self.crop:
            i, j, h, w = T.RandomCrop.get_params(image, self.crop)
            image, mask = F.crop(image, i, j, h, w), F.crop(mask, i, j, h, w)

        if np.random.rand() < self.p_flip:
            image, mask = F.hflip(image), F.hflip(mask)

        # color transforms || ONLY ON IMAGE
        if self.color_jitter_params:
            image = self.color_tf(image)

        # random affine transform
        if np.random.rand() < self.p_random_affine:
            # affine_params = T.RandomAffine(180).get_params((-90, 90), (1, 1), (2, 2), (-45, 45), self.crop)
            # affine_params = T.RandomAffine(180).get_params((-90, 90), (0.7, 1.5), (0.7, 1.5), (-45, 45), self.crop)
            # affine_params = T.RandomAffine(degrees=40, scale=(.9, 1.1), shear=0).get_params((-90, 90), (0.9, 1.1), (0.9, 1.1), (-15, 15), self.crop)
            # affine_params = T.RandomAffine(degrees=40, scale=(.9, 1.1), shear=0).get_params((-90, 90), (1, 1), (2, 2), (-45, 45), self.crop)
            affine_params = T.RandomAffine(degrees=90, scale=(.8, 1.2), shear=0).get_params((-90, 90), (0.5, 1), (1, 2), (-15, 15), self.crop)
            image, mask = F.affine(image, *affine_params), F.affine(mask, *affine_params)

        cvret, cvlabels, cvstats, cvcentroid = cv2.connectedComponentsWithStats(np.asarray(mask))
        labelsSemantic = list(map(lambda x:x[:4], filter(lambda x: x[4]<10000, cvstats)))

        # transforming to tensor
        image = F.to_tensor(image)
        if not self.long_mask:
            mask = F.to_tensor(mask)
        else:
            mask = to_long_tensor(mask)

        if self.normTransform:
            image = self.normTransform(image)

        return image, mask, labelsSemantic

class ImageToImage2DTest(Dataset):
    def __init__(self, cropMode='ori', crop=None, maxTargetNum=50):
        if cropMode == 'ori':
            self.crop = None
        elif cropMode == 'fixed':
            self.crop = 512 if crop is None else crop
        # self.dataset = ImageTeadDataset('/home/hit/Program/images/test/img',
        #                                 '/home/hit/Program/images/test/mask')
        self.dataset = ImageTeadDataset(r'Z:/data/train/new_img',
                                        r'Z:/data/train/new_mask')
        self.maxTargetNum = maxTargetNum
        self.long_mask = True

    def __getitem__(self, index):
        image, mask, labelsExample = self.dataset.__getitem__(index)
        # assert image.shape == mask.shape
        
        image, meta = processGray(image, scale=1., inp_h=self.crop, inp_w=self.crop)
        mask, _, = processGray(mask, scale=1., inp_h=self.crop, inp_w=self.crop)
        image, mask = F.to_pil_image(image), F.to_pil_image(mask)
        image = F.to_tensor(image)
        if not self.long_mask:
            mask = F.to_tensor(mask)
        else:
            mask = to_long_tensor(mask)

        coors = torch.zeros((self.maxTargetNum, 5))
        assert len(labelsExample) <= self.maxTargetNum
        for i, val in enumerate(labelsExample):
            coors[i, 0] = 1.
            coors[i, 1:] = torch.tensor(labelsExample[i])

        tran_inv = meta['trans_input_inv']
        tran_inv = torch.tensor(tran_inv, dtype=torch.float32)

        # shape = [meta['in_width'], meta['in_height']]
        # shape = torch.tensor(shape, dtype=torch.float32)

        imageSize = meta['in_width'] * meta['in_height']

        return image, mask, coors, tran_inv, imageSize

    def __len__(self):
        return len(self.dataset)

class ImageToImage2D(Dataset):
    def __init__(self, crop_size=512, heatmap=True, maxTargetNum=50, datasetMode='read'):
        self.crop_size = crop_size
        if datasetMode == 'create':
            self.dataset = ImageCreatorDataset()
        elif datasetMode == 'read':
            self.dataset = ImageTeadDataset(r'Z:/data/train/new_img',
                                            r'Z:/data/train/new_mask')
        # self.joint_transform = JointTransform2D(crop=crop_size, p_flip=0.5, color_jitter_params=(0.2, 0.2, 0.3, 0.3), normTransform=(0, 1), long_mask=True, p_random_affine=0.6)
        # self.joint_transform = JointTransform2D(crop=(crop_size, crop_size), p_flip=0.5,
        #                                         color_jitter_params=(0.2, 0.2, 0.3, 0.3), normTransform=(0, 1),
        #                                         long_mask=True, p_random_affine=0.6)
        self.joint_transform = JointTransform2D(crop=(crop_size, crop_size), p_flip=0.5,
                                                color_jitter_params=(0.2, 0.2, 0.3, 0.3), normTransform=(0, 1),
                                                long_mask=True, p_random_affine=-1.)
        self.heatmap = heatmap
        self.maxTargetNum = maxTargetNum

    def __getitem__(self, index):
        image, mask, labelsExample = self.dataset.__getitem__(index)
        h, w = image.shape[:2]
        if self.crop_size != 0:
            auxLev = max(self.crop_size - h, 0)
            auxCow = max(self.crop_size - w, 0)
            ranL = random.randint(0, auxLev)
            ranC = random.randint(0, auxCow)
            image = np.pad(image, ((ranL, auxLev-ranL), (ranC, auxCow-ranC)), constant_values=0)
            mask = np.pad(mask, ((ranL, auxLev-ranL), (ranC, auxCow-ranC)), constant_values=0)
            # print("............")
            # print(image.shape)

        # correct dimensions if needed
        image, mask = correct_dims(image, mask)
        # print(image.shape)
        mask[mask<127] = 0
        mask[mask>=127] = 1

        image, mask, labelsSemantic = self.joint_transform(image, mask)

        if self.heatmap:
            heatmap = np.zeros((self.crop_size, self.crop_size))
            for xt, yt, wt, ht in labelsSemantic:
                cx = xt + wt/2
                cy = yt + ht/2
                draw_umich_gaussian(heatmap, (cx, cy), 3)
            heatmap = torch.tensor(heatmap, dtype=torch.float32).unsqueeze(0)

        # ########### STA DEBUG ###########
        # auxHM = np.uint8(heatmap[0].numpy()*255)
        # auxImg = np.uint8(image[0].numpy()*255)
        # cv2.imshow('auxHM', auxHM)
        # cv2.imshow('auxImg', auxImg)
        # cv2.waitKey(0)
        # ########### END DEBUG ###########

        # return image, heatmap if self.heatmap else mask

        coors = torch.zeros((self.maxTargetNum, 5))
        assert len(labelsSemantic) <= self.maxTargetNum
        for i, val in enumerate(labelsSemantic):
            coors[i, 0] = 1.
            coors[i, 1:] = torch.tensor(labelsSemantic[i])

        # mask = torch.tensor(mask, dtype=torch.float32)
        mask = mask.float().unsqueeze(0)
        # return image, heatmap if self.heatmap else mask, coors
        return image, heatmap, mask, coors

    def __len__(self):
        return len(self.dataset)

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    # 限制最小的值
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    # 一个圆对应内切正方形的高斯分布

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius +
                               bottom, radius - left:radius + right]

    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        # 将高斯分布覆盖到heatmap上，取最大，而不是叠加
    return heatmap

def _nms(heat, kernel=3):
    hmax = nnF.max_pool2d(heat, kernel, stride=1, padding=(kernel - 1) // 2)
    keep = (hmax == heat).float()
    return heat * keep

def _gather_feature(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _topk(scores, K=50):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feature(topk_inds.view(
        batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feature(topk_ys.view(
        batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feature(topk_xs.view(
        batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

def decode(heatmap, k=50):
    heatmap = _nms(heatmap)
    scores, inds, clses, ys, xs = _topk(heatmap, K=k)
    x = xs.unsqueeze(-1)
    coors = torch.cat((xs.unsqueeze(-1), ys.unsqueeze(-1), scores.unsqueeze(-1)), dim=2)
    return coors

def decodeTest(heatmap, pt):
    coors = decode(heatmap)
    ret = coors.clone()
    ret[:, :, 2] = 1.
    coors[:, :, :2] = torch.einsum('ijk, ilk -> ijl', ret, pt) # 1,50,3 1,2,3 -> 1,50,2
    # print(coors[0,:2])
    # for i, coor in enumerate(coors):
    #     coors[i, :, :2] = torch.mm(ret[i], pt[i].T) # 50,3 3,2 -> 50,2
    # print(coors[0,:2])
    return coors

def affine_transform(pt, t):
  new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
  new_pt = np.dot(t, new_pt)
  return new_pt[:2]
