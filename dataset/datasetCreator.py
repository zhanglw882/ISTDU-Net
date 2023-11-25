import glob, os
import cv2
import numpy as np
import random

from .clone import imageClone
from .gaus import augGaus

# 在mask黑里，通过，白里，一定概率重来
def chooseCoor(backImg):
    kernel = np.ones((5, 5), np.uint8)
    h, w = backImg.shape
    prob = 0.95
    # prob = 2
    th, ret = cv2.threshold(backImg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret = 255 - ret
    cvret, cvlabels, cvstats, cvcentroid = cv2.connectedComponentsWithStats(ret)
    idx = np.argmax([np.sum(cvlabels[cvlabels==i]) for i in range(cvret)])
    ret = np.zeros((h, w), dtype=np.uint8)
    ret[cvlabels==idx] = 255
    ret = cv2.erode(ret, kernel)
    # cv2.imshow('bgmask', ret)
    count = 0
    while True:
        count += 1
        x, y = random.randint(10, w - 10), random.randint(10, h - 10)
        if ret[y, x] == 255 or random.random() > prob or count > 100:
            break
    return (x, y)

# 用于返回随机背景idx，目标属性s, 包括(tarIdxs, tarCoors, tarSizes, tarLevels)*tarNum
# tarCoorModewei为random或者other
class RandomController:
    def __init__(self, lenBack, lenTar, maxTarNum=10, tarSizeRange=(1, 15), tarLWRatio=(0.66, 1.5),
                 tarLevelRange=(10, 50), backRandom=False, backAugFactorRange=(0.66, 1.5),
                 tarCoorMode='random', backAugGauss=True, gaussBlur='True'):
        self.backRandom = backRandom
        self.lenBack = lenBack
        self.lenTar = lenTar
        self.maxTarNum = maxTarNum
        self.tarSizeRange = tarSizeRange
        self.tarLWRatio = tarLWRatio
        self.tarLevelRange = tarLevelRange
        self.backAugFactorRange = backAugFactorRange
        self.backAugGauss = backAugGauss
        self.gaussBlur = gaussBlur

        if tarCoorMode == 'random':
            self.getTarCoor = self.getTarCoorRandom
        elif tarCoorMode == 'other':
            self.getTarCoor = chooseCoor

    def getBackIdx(self, idx):
        return idx if self.backRandom==False else random.randint(0, self.lenBack-1)

    def getTarProps(self, backImg=None, tarNum=None):
        if tarNum == None:
            tarNum = random.randint(0, self.maxTarNum)
        tarProps = []
        for i in range(tarNum):
            tarIdx = random.randint(0, self.lenTar-1)

            # tarCoor = (random.randint(0, w-1), random.randint(0, h-1))
            # tarCoor = (random.randint(5, w-5), random.randint(5, h-5))
            # tarCoor = self.getTarCoor(backImg.shape[::-1])
            tarCoor = self.getTarCoor(backImg)

            auxW = random.randint(*self.tarSizeRange)
            tarSize = (auxW, max(1, int(auxW*random.uniform(*self.tarLWRatio))))
            tarLevel = random.randint(*self.tarLevelRange)
            tarProps.append([tarIdx, tarCoor, tarSize, tarLevel])
        return tarProps

    def getBackAugFactor(self):
        return random.uniform(*self.backAugFactorRange) if self.backAugFactorRange is not None else 1

    def getTarCoorRandom(self, img):
        h, w = img.shape
        return (random.randint(10, w-10), random.randint(10, h-10))

    def getBackAugGauss(self):
        return self.backAugGauss

    def getGaussBlur(self):
        if self.gaussBlur == 'True':
            return True
        elif self.gaussBlur == 'False':
            return False
        elif self.gaussBlur == 'random':
            return random.random() < 0.5

class Aug:
    # def __init__(self, tarFlip=False, backFactor=True, augO=True):
    def __init__(self, tarFlip=False, backFactor=True):
        self.tarFlip = tarFlip
        self.backFactor = backFactor
        # self.augO = augO

    def tarAug(self, tarImg, tarSize):
        tarImg = cv2.resize(tarImg, tarSize)
        return tarImg

    def backAug(self, backImg, factor, augO=False):
        if self.backFactor:
            backImg = np.uint8(np.clip(backImg * factor, 0, 255))
            backImg = augGaus(backImg) if augO else backImg
        return backImg

class Creator:
    def __init__(self, backgrounds=None, targets=None, mergeMode='add', aug=None):
        self.backgrounds = sorted(glob.glob(os.path.join(backgrounds, '*.*g')))
        self.targets = sorted(glob.glob(os.path.join(targets, '*.*g')))
        if mergeMode == 'add':
            self.calPixMax = self.calPixMaxAdd
        elif mergeMode == 'change':
            self.calPixMax = self.calPixMaxChange
        self.aug = Aug()
        self.image = None
        # self.blur = True

    # backIdx 范围为 0~lenBack() tarProp 包括tarIdxs, tarCoors, tarSizes, tarLevels
    def create(self, tarProps, gaussBlur):
        mask = np.zeros(self.image.shape, dtype=np.uint8)
        labelsExample = []
        for tarIdx, tarCoor, tarSize, tarLevel in tarProps:
            tarImg = cv2.imread(self.targets[tarIdx], cv2.IMREAD_GRAYSCALE)
            tarImg = self.aug.tarAug(tarImg, tarSize)
            pixMax = self.calPixMax(tarLevel, tarCoor)
            ok, auxImg, auxMask = imageClone(tarImg, self.image, tarCoor, pixMax=pixMax, OSTU=False, Gauss=False)
            auxMask = np.uint8(auxMask*255)
            cvret, cvlabels, cvstats, cvcentroid = cv2.connectedComponentsWithStats(auxMask)
            labelsExample += list(map(lambda x:x[:4], filter(lambda x: x[4]<10000, cvstats)))
            self.image = np.max(np.array([auxImg, self.image]), axis=0)
            mask = cv2.bitwise_or(mask, auxMask)
        cvret, cvlabels, cvstats, cvcentroid = cv2.connectedComponentsWithStats(mask)
        labelsSemantic = list(map(lambda x:x[:4], filter(lambda x: x[4]<10000, cvstats)))
        if gaussBlur:
            self.image = cv2.GaussianBlur(self.image, (3, 3), 0)
        return self.image, mask, labelsSemantic, labelsExample

    def calPixMaxAdd(self, tarLevels, tarCoor):
        return tarLevels + self.image[tarCoor[::-1]]

    def calPixMaxChange(self, tarLevels, _):
        return tarLevels

    def loadBackImage(self, idx, backAugFactor, backAugGauss):
        self.image = cv2.imread(self.backgrounds[idx], cv2.IMREAD_GRAYSCALE)
        self.image = self.aug.backAug(self.image, backAugFactor, backAugGauss)
        return self.image

    def lenBack(self):
        return len(self.backgrounds)

    def lenTar(self):
        return len(self.targets)