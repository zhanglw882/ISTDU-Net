import os
import glob
import cv2

from torch.utils.data import Dataset
from .datasetCreator import Creator, RandomController

class ImageTeadDataset(Dataset):
    def __init__(self, imgPath=None, labelPath=None):
        if imgPath is None:
            # imgPath = '/home/hit/Program/gitMycode/local/Creator/dataset/splitCreate/images/testBack/inpaintedImages'
            imgPath = '/home/zhang/ZLW/Small_Target_Detect/Infrared_SmallTargetDetection/Infrared_small_target_detector_all/open-alcnet/data/train/img'
        if labelPath is None:
            # labelPath = '/home/hit/Program/gitMycode/local/Creator/dataset/splitCreate/images/testBack/masks'
            labelPath = '/home/zhang/ZLW/Small_Target_Detect/Infrared_SmallTargetDetection/Infrared_small_target_detector_all/open-alcnet/data/train/mask'
        self.images = glob.glob(os.path.join(imgPath, '*.*'))
        self.images = list(filter(lambda x: os.path.basename(x).split('.')[1] in ['jpg', 'png', 'bmp'], self.images))
        self.images.sort()
        self.labels = glob.glob(os.path.join(labelPath, '*.*'))
        self.labels = list(filter(lambda x: os.path.basename(x).split('.')[1] in ['jpg', 'png', 'bmp'], self.labels))
        self.labels.sort()

        assert len(self.labels) == len(self.images)

    def __getitem__(self, idx):
        image = cv2.imread(self.images[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(self.labels[idx], cv2.IMREAD_GRAYSCALE)
        assert image.shape == mask.shape
        cvret, cvlabels, cvstats, cvcentroid = cv2.connectedComponentsWithStats(mask)
        labelsExample = list(map(lambda x:x[:4], filter(lambda x: x[4]<10000, cvstats)))
        return image, mask, labelsExample

    def __len__(self):
        return len(self.images)

class ImageCreatorDataset(Dataset):
    def __init__(self, mode='random', backgrounds=None, targets=None, config=None):
        if backgrounds is None:
            backgrounds = '/home/hit/Program/images/lir/trainBack'
        if targets is None:
            targets = '/home/hit/Program/gitMycode/local/Creator/Single/tar/img'
        self.creator = Creator(backgrounds, targets)
        self.mode = mode
        if self.mode == 'random':
            if config is None:
                config = {'lenBack': self.creator.lenBack(),
                          'lenTar': self.creator.lenTar(),
                          'maxTarNum': 10,
                          'tarSizeRange': (1, 15),
                          'tarLWRatio' :(0.4, 1.),
                          'tarLevelRange': (7, 50),
                          'backRandom': False,
                          'backAugFactorRange':(0.8, 1.2),
                          'tarCoorMode': 'other',
                          'backAugGauss': True,
                          'gaussBlur': 'random'}
            self.config = RandomController(**config)
        self.len = self.creator.lenBack()

    def __getitem__(self, idx):
        if self.mode == 'random':
            backIdx = self.config.getBackIdx(idx)
            auxBack = self.creator.loadBackImage(backIdx, self.config.getBackAugFactor(), self.config.getBackAugGauss())
            tarProps = self.config.getTarProps(auxBack)
            image, mask, labelsSemantic, labelsExample = self.creator.create(tarProps=tarProps, gaussBlur=self.config.getGaussBlur())
        return image, mask, labelsExample

    def __len__(self):
        return self.len

if __name__ == '__main__':
    import cv2
    ds = ImageCreatorDataset()
    # ds = ImageTeadDataset()
    for image, mask, labelsExample, in ds:
        rect = image.copy()
        cv2.imshow('image', image)
        cv2.imshow('mask', mask)
        # print(labelsExample)
        for coor in labelsExample:
            cv2.rectangle(rect, tuple(coor[:2]), tuple(map(sum, zip(coor[:2], coor[2:]))), color=255, thickness=1)
        cv2.imshow('rect', rect)
        cv2.waitKey(0)
