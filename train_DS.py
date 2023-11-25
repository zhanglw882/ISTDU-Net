import torch
import torch.nn as nn
from torch.utils import data
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import numpy as np
import os
import time

from utils import RAdam, LR_Scheduler
from dataset import ImageToImage2D, decode, ImageToImage2DTest, decodeTest
from torchtoolbox.optimizer import Lookahead
from tqdm import tqdm
# from tensorboardX import SummaryWriter

from eval import Evaluator

from model import Network, Loss
from tensorboardX import SummaryWriter
class Trainer():
    def __init__(self):
        self.loadPath = None
        self.savePath = './save_pth/ISTDU_Net/'
        if not os.path.exists(self.savePath):
            os.makedirs(self.savePath)
        self.base_lr = 1e-3
        # self.base_lr = 1
        self.moment = 0.9
        self.wd = 1e-4
        self.maxepoch = 12000
        # self.batch = 48
        self.batch = 2
        self.batch_val = 1
        # self.num_class = 2
        self.WR_EPOCH = 50
        self.cuda = True
        # self.num_workers = 16
        # self.num_workers = 1
        self.num_workers = 12
        # self.base_size = 280
        self.crop_size = 512

        # self.root = ''
        self.lr_mode = 'cos'

        self.datasets = ImageToImage2D(crop_size=self.crop_size)
        self.datasets_val = ImageToImage2DTest()
        self.trainloader = data.DataLoader(dataset=self.datasets, num_workers=self.num_workers, batch_size=self.batch, drop_last=True, shuffle=True)
        self.valloader = data.DataLoader(dataset=self.datasets_val, num_workers=self.num_workers,batch_size=self.batch_val, drop_last=False, shuffle=False)
        self.bins = 100
        # trainDecodeConf = {'k': 50}
        # trainDecodeTestConf = {'k': 50}
        self.Eval = Evaluator(decode, bins=self.bins, device='cuda' if self.cuda else 'cpu')
        self.Eval_val = Evaluator(decodeTest, bins=self.bins, device='cuda' if self.cuda else 'cpu')

        self.model = Network()
        self.loss_layer = Loss()

        params_list = [{'params': self.model.parameters(), 'lr': self.base_lr}, ]
        # params_list = [{'params': self.model.encoder.parameters(), 'lr': self.base_lr}, ]
        # if hasattr(self.model, 'decoder'):
        #     params_list.append({'params': self.model.decoder.parameters(), 'lr': self.base_lr * 10})

        # self.optimizer = optim.SGD(params_list, lr=self.base_lr, momentum=self.moment, weight_decay=self.wd)
        self.optimizer = RAdam(params_list,lr=self.base_lr,weight_decay=self.wd)
        self.optimizer = Lookahead(self.optimizer)
        self.scheduler = LR_Scheduler(mode=self.lr_mode, base_lr=self.base_lr, num_epochs=self.maxepoch,
                                      iters_per_epoch=len(self.trainloader), warmup_epochs=self.WR_EPOCH)
        # self.model.init_weight()
        if self.cuda:
            cudnn.benchmark = True
            # self.model = self.model.cuda()
            self.model = nn.DataParallel(self.model, device_ids=[i for i in range(torch.cuda.device_count())]).cuda()
            # self.model = nn.DataParallel(self.model, device_ids=[0,1,2]).cuda()
            self.loss_layer = self.loss_layer.cuda()

        if self.loadPath:
            self.model.load_state_dict(torch.load(self.loadPath))

        print(len(self.trainloader))

        self.best = 0.0
        self.summary_writer = SummaryWriter(log_dir='./log/')

        pytorch_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Total_params: {}".format(pytorch_total_params))

    def train(self, epoch, loss_list, metrics_list):
        train_loss = 0.0
        self.model.train()
        torch.cuda.synchronize()
        t0 = time.time()
        for i, (image, label, mask, coors) in tqdm(enumerate(self.trainloader), total=len(self.trainloader)):
            # break
            # torch.cuda.synchronize()
            # t1 = time.time()

            if self.cuda:
                im = Variable(image).cuda()
                la = Variable(label).cuda()
                ma = Variable(mask).cuda()
                co = Variable(coors).cuda()
            else:
                im = Variable(image)
                la = Variable(label)
                ma = Variable(mask)
                co = Variable(coors)
            self.scheduler(self.optimizer, i, epoch)
            self.optimizer.zero_grad()
            outD, outS = self.model(im)
            loss = self.loss_layer(outD, la) + self.loss_layer(outS, ma)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            self.summary_writer.add_scalar('Loss', loss.item(), i+epoch*len(self.trainloader))

            # preds = self.get_pred(outs).cuda()
            self.Eval.add_batch(outD, co, [self.crop_size*self.crop_size]*image.shape[0])

            # self.Eval.add_batch(label, torch.argmax(outs[-1].data, dim=1))

            # self.Eval.add_batch(label,preds.data)
            # torch.cuda.synchronize()
            # print(time.time() - t1,loss.item())

        val_metrics = self.val()

        _, _, metrics = self.Eval.metrics()
        self.Eval.reset()
        metrics = float(metrics)
        torch.cuda.synchronize()

        print('cost:{}, epoch:{}, loss:{}, metrics:{}, val_metrics:{}'
              .format(time.time() - t0, epoch,train_loss / i, metrics, val_metrics))
        # print('cost:{} ,epoch:{} ,loss:{}'
        #       .format(time.time() - t0, epoch, train_loss / i))
        # if (epoch % 5 == 0 and epoch != 0):

        #
        # if (epoch % 5 == 0 and epoch <50 and epoch != 0) or (epoch >= 50):
        #     torch.save(self.model.state_dict(), self.savePath + 'epoch_' + repr(epoch) + '.pth')
        if val_metrics > self.best:
            self.best = val_metrics
            torch.save(self.model.state_dict(), self.savePath + 'best_new.pth')

        loss_list.append(train_loss / i)
        # metrics_list.append(metrics)

    def val(self):
        self.model.eval()
        for i, (image, label, coors, trans_input_inv, imageSize) in enumerate(self.valloader):
            if self.cuda:
                im = Variable(image).cuda()
                la = Variable(label).cuda()
                co = Variable(coors).cuda()
                trans_input_inv = trans_input_inv.cuda()
            else:
                im = Variable(image)
                la = Variable(label)
                co = Variable(coors)
            with torch.no_grad():
                outs, _ = self.model(im)
            self.Eval_val.add_batch(outs, co, imageSize, trans_input_inv)

        _, _, val_metrics = self.Eval_val.metrics()
        self.Eval_val.reset()
        return float(val_metrics)

if __name__ == '__main__':
    trainer = Trainer()
    loss_list = list()
    metrics_list = list()
    for ep in range(0, trainer.maxepoch):
        torch.cuda.empty_cache()
        # trainer.val()
        trainer.train(ep,loss_list, metrics_list)
    torch.save(trainer.model.state_dict(), trainer.savePath + 'last.pth')
    trainer.summary_writer.close()
    from matplotlib import pyplot as plt
    plt.plot(range(len(loss_list)), loss_list, color='black')
    # plt.plot(range(len(metrics_list)), loss_list, color='red')
    plt.savefig('./res.jpg')