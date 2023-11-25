import torch
import numpy as np

# if labelMode == 'mask':
#     self.evaluator = EvaluatorMask()
# elif labelMode == 'example':
#     self.evaluator = EvaluatorExam()

class Evaluator:
    def __init__(self, pocessing = lambda x:x, bins=10, device='cpu'):
        self.pocessing = pocessing
        # self.record = []

        self.device = device
        self.bins = bins
        self.reset()
        # self.tp_arr = torch.zeros(self.bins+1)
        # self.pos_arr = torch.zeros(self.bins+1)
        # self.numTar = 0
        # self.numBack = 0

        self.maxDist = 100

    def metrics(self):
        tp_arr = self.tp_arr.clone()
        fp_arr = self.fp_arr.clone()
        # for i in range(self.bins-2, -1, -1):
        #     self.tp_arr[i] += self.tp_arr[i+1]
        #     self.fp_arr[i] += self.fp_arr[i+1]
        # tpr = self.tp_arr / self.numTar
        # fpr = self.fp_arr / self.numBack
        # for i in range(self.bins-2, -1, -1):
        for i in range(self.bins-1, -1, -1):
            tp_arr[i] += tp_arr[i+1]
            fp_arr[i] += fp_arr[i+1]
        # for i in range(1, self.bins+1):
        #     fp_arr[i] += fp_arr[i-1]
        tpr = tp_arr / self.numTar
        fpr = fp_arr / self.numBack

        # tpr = tpr.numpy()
        # fpr = fpr.numpy()


        # 计算AUC
        auc = 0.
        for i in range(self.bins):
            auc += (tpr[i] + tpr[i+1]) * (fpr[i] - fpr[i+1]) / 2.
        auc += tpr[-1] * fpr[-1] / 2
        auc += (1 + tpr[0]) * (1 - fpr[0]) / 2

        # leftTpr = 1.
        # rightFpr = 1.
        # auc = 0.
        # for i in range(self.bins+1):
        #     # print(i)
        #     auc += (leftTpr + tpr[i]) * (rightFpr - fpr[i]) / 2
        #     leftTpr = tpr[i]
        #     rightFpr = fpr[i]
        # auc += (leftTpr + 0) * (rightFpr - 0) / 2

        if auc > 1:
            # ########### STA DEBUG ###########
            import matplotlib.pyplot as plt
            plt.plot(fpr.detach().cpu().numpy(), tpr.detach().cpu().numpy())
            plt.show()
            # ########### END DEBUG ###########
            print(int(auc))
        return tpr, fpr, auc

    def add_batch(self, predsB, labelsB, imageSize, kwargs=None): # 哪有这么用的
        batchSize = predsB.shape[0]
        if kwargs is not None:
            predsB = self.pocessing(predsB, kwargs)
        else:
            predsB = self.pocessing(predsB)
        for b in range(batchSize):
            labels = labelsB[b]
            lenL = int(torch.sum(labels[:, 0]))
            # if lenL:
            labels = labels[:lenL, 1:]
            preds = predsB[b]
            # predConf = preds[:, 4]
            predConf = preds[:, -1]
            if lenL:
                # matrix = torch.zeros((len(labels), len(preds))).to(self.device)
                matrix = torch.zeros((len(preds), len(labels))).to(self.device)
                for i, label in enumerate(labels):
                    for j, pred in enumerate(preds):
                        matrix[j, i] = self.calDist(pred, label)
                mask = matrix <= self.maxDist
                maskConf = predConf.unsqueeze(-1) * mask
                confForLabels = maskConf.clone()
                confForLabels[confForLabels==0.] = -1. # 什么弯弯绕绕的奇怪写法？
                # confForLabels = torch.min(confForLabels, dim=0).values
                confForLabels = torch.max(confForLabels, dim=0).values
                # confForLabels[confForLabels==2.] = 2.
                tpIdx = confForLabels * self.bins
                for idx in tpIdx:
                    if idx > self.bins or idx < 0:
                        continue
                    self.tp_arr[int(idx)] += 1

            maskForPreds = torch.sum(mask, dim=1) > 0 if lenL else torch.zeros(len(preds), dtype=torch.bool).to(self.device)
            fpIdx = predConf * self.bins
            for idx, cont, in zip(fpIdx, maskForPreds):
                if cont:
                    continue
                self.fp_arr[int(idx)] += 1

            self.numTar += lenL
            self.numBack += imageSize[b]

    # def calDist(self, pred, label):
    #     return abs(pred[0] + pred[2]/2 - label[0] - label[2]/2) \
    #            + abs(pred[1] + pred[3]/2 - label[1] - label[3]/2)

    def calDist(self, pred, label):
        return (pred[0] - label[0] - label[2]/2)**2 \
               + (pred[1] - label[1] - label[3]/2)**2

    def reset(self):
        self.tp_arr = torch.zeros(self.bins+1).to(self.device)
        # self.tp_arr[-1] = 1.
        self.fp_arr = torch.zeros(self.bins+1).to(self.device)
        # self.fp_arr[-1] = 1.
        self.numTar = 0
        self.numBack = 0

# class EvaluatorMask(Evaluator):
#     def __init__(self):
#
#     def

# class EvaluatorMask(Evaluator):
#     def __init__(self):

    # def add_batch(self, pred, label):
    #     pass


if __name__ == '__main__':
    # pred = [[[10,10,5,5,0.5],[20,20,5,5,0.4]],[[10,10,5,5,0.6],[10,10,5,5,0.8]]]
    pred = [[[15., 15., 0.8], [16., 17., 0.2], [100., 100., 0.9]]]
    pred = torch.tensor(pred)
    # print(pred)
    # label = [[[1,10,10,5,5],[1,20,20,5,5],[0,-1,-1,-1,-1]],[[1,10,10,5,5],[1,20,20,5,5],[0,-1,-1,-1,-1]]]
    label = [[[1., 200., 200., 10., 10.], [1., 10., 10., 10., 10.], [0., 0., 0., 0., 0.]]]
    # label = [[[0., 200., 200., 10., 10.], [0., 10., 10., 10., 10.], [0., 0., 0., 0., 0.]]]
    label = torch.tensor(label)

    bins = 10
    e = Evaluator(bins=bins)
    e.add_batch(pred, label, imageSize=512*512)
    tpr, fpr, auc = e.metrics()
    print(float(auc))
    import matplotlib.pyplot as plt

    tpr = tpr.numpy()
    fpr = fpr.numpy()
    for i, j in zip(tpr, fpr):
        plt.scatter(j, i)
        print(j, i)
    # plt.scatter(1., 1.)
    # plt.plot(fpr, tpr)
    # plt.plot(tpr, fpr)
    plt.show()
