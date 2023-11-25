from torch.utils.data import Dataset

from datasetCreator import Creator, RandomController

class ImageCreatorDataset(Dataset):
    def __init__(self, mode='random', backgrounds=None, targets=None, config=None):
        if backgrounds is None:
            backgrounds = '/home/hit/Program/MyProgram/ThermalImagerLP/ChooseImage/savePath'
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
                          'tarLWRatio' :(0.66, 1.5),
                          'tarLevelRange': (10, 50),
                          'backRandom': False,
                          'backAugFactorRange': (0.66, 1.35),
                          'tarCoorMode': 'random',
                          'backAugGauss': True}
            else:
                config['lenBack'] = self.creator.lenBack()
                config['lenTar'] = self.creator.lenTar()
            self.config = RandomController(**config)
        self.len = self.creator.lenBack()

    def __getitem__(self, idx):
        if self.mode == 'random':
            backIdx = self.config.getBackIdx(idx)
            auxBack = self.creator.loadBackImage(backIdx, self.config.getBackAugFactor(), self.config.getBackAugGauss())
            tarProps = self.config.getTarProps(auxBack)
            image, mask, labelsSemantic, labelsExample, = self.creator.create(tarProps=tarProps)
        return image, mask, labelsExample

    def __len__(self):
        return self.len

if __name__ == '__main__':
    import cv2
    config = {'lenBack': None,
              'lenTar': None,
              'maxTarNum': 5,
              'tarSizeRange': (1, 15),
              'tarLWRatio': (0.2, 1.5),
              'tarLevelRange': (15, 50),
              'backRandom': True,
              # 'backAugFactorRange': (0.8, 1.2),
              'backAugFactorRange': None,
              'tarCoorMode': 'other',
              # 'tarCoorMode': 'random',
              'backAugGauss': False}
    ds = ImageCreatorDataset(config=config)
    # ds = ImageCreatorDataset()
    # for image, mask, labelsExample in ds:
    for idx in range(len(ds)):
        image, mask, labelsExample = ds.__getitem__(idx)
        rect = image.copy()
        cv2.imshow('image', image)
        cv2.imshow('mask', mask)
        # print(labelsExample)
        for coor in labelsExample:
            cv2.rectangle(rect, tuple(coor[:2]), tuple(map(sum, zip(coor[:2], coor[2:]))), color=255, thickness=1)
        cv2.imshow('rect', rect)
        cv2.waitKey(0)
