import numpy as np
import torch.nn as nn
from collections import Counter
import os
import pickle as dill
import torch.optim as optim
import torch
import torch.nn.functional as F
from src import Glo
from src.FocalLoss import FocalLoss
from src.util import record_logs,write_res,train,test,preparing_data_for_training, write_tensorboard
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler
import math


def KL(alpha, c):
    beta = torch.ones((1, c)).cuda()
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


def ce_loss(p, alpha, c, global_step, annealing_step):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1

    label = p

    A = torch.sum(label * (torch.digamma(S).expand_as(alpha) - torch.digamma(alpha)), dim=1, keepdim=True)

    annealing_coef = min(1.0, global_step / annealing_step)

    alp = E * (1 - label) + 1
    B = annealing_coef * KL(alp, c)

    return (A + B)


class Config(object):
    def __init__(self):
        # self.conv_subsample_lengths = [1, 2, 1, 2, 1, 2, 1, 2]
        self.conv_subsample_lengths = [1,2,1,2,1,2,1]
        self.conv_filter_length = 64
        self.conv_num_filters_start = 12
        self.conv_dropout = 0.5
        self.conv_num_skip = 2
        self.conv_increase_channels_at = 2

class ZeroPad1d(nn.Module):
    def forward(self, x):
        pad = torch.zeros_like(x)
        return torch.cat([x, pad], dim=1)

class ResNetBlock(nn.Module):
    def __init__(self, cur_num_filters, prior_num_filters,subsample_length, block_index, config):
        super(ResNetBlock, self).__init__()

        self.maxpool = nn.MaxPool1d(kernel_size=subsample_length)
        self.zeropad = ZeroPad1d()
        self.zero_pad = (block_index % config.conv_increase_channels_at) == 0 and block_index > 0
        self.rnb = self._make_rnb(block_index,cur_num_filters, prior_num_filters, subsample_length, config)

    def _same_pad(self, x, conv):
        in_length = x.shape[-1]
        kernel_size = conv.kernel_size[0]
        stride = 1

        padding_needed = max(0, (in_length - 1) * stride + kernel_size - in_length)
        pad_left = padding_needed // 2
        pad_right = padding_needed - pad_left

        x = F.pad(x, (pad_left, pad_right))
        return x

    def _make_rnb(self, block_index,  cur_num_filter, prior_num_filters, subsample_length, config):
        layers = []
        isConvExist = False
        for i in range(config.conv_num_skip):
            if not (block_index == 0 and i == 0):
                if isConvExist:
                    num_filters = cur_num_filter
                else:
                    num_filters = prior_num_filters
                layers.append(nn.BatchNorm1d(num_filters))
                layers.append(nn.ReLU(inplace=True))
                if i>0:
                    layers.append(nn.Dropout(config.conv_dropout))

            if isConvExist:
                inChannels = cur_num_filter
                outChannels = cur_num_filter
            else:
                inChannels = prior_num_filters
                outChannels = cur_num_filter

            conv_layer = nn.Conv1d(
                in_channels=inChannels,
                out_channels=outChannels,
                kernel_size=config.conv_filter_length,
                stride=subsample_length if i == 0 else 1,
            )
            layers.append(conv_layer)
            isConvExist = True
        return nn.Sequential(*layers)

    def forward(self, x):
        shortcut = self.maxpool(x)
        if self.zero_pad:
            shortcut = self.zeropad(shortcut)
        for layer in self.rnb:
            if isinstance(layer, nn.Conv1d):
                x = self._same_pad(x, layer)
            x = layer(x)
        x = x + shortcut
        return x

def get_num_filters_at_index(index, num_start_filters, config):
    return 2**int(index / config.conv_increase_channels_at) * num_start_filters

class ResNet(nn.Module):
    def __init__(self,input_size,dim, num_classes):
        super(ResNet, self).__init__()

        self.config = Config()
        self.relu = nn.LeakyReLU()

        self.conv1 = nn.Conv1d(input_size, self.config.conv_num_filters_start, kernel_size=self.config.conv_filter_length, padding='same',stride=1)
        self.bn1 = nn.BatchNorm1d(self.config.conv_num_filters_start)

        self.rnbs = self.make_layers()

        self.pool = torch.nn.AdaptiveAvgPool1d(1)

        index_last_subsample_length = len(self.config.conv_subsample_lengths)-1
        fc_in = get_num_filters_at_index(index_last_subsample_length, self.config.conv_num_filters_start, self.config)
        self.fc = nn.Linear(fc_in, num_classes)

        self.logits2evidence = nn.LeakyReLU()

    def make_layers(self):
        layers = []
        for index, subsample_length in enumerate(self.config.conv_subsample_lengths):
    
            cur_num_filters = get_num_filters_at_index(index, self.config.conv_num_filters_start, self.config)
            if index == 0:
                pri_num_filters = get_num_filters_at_index(0, self.config.conv_num_filters_start, self.config)
            else:
                pri_num_filters = get_num_filters_at_index(index-1, self.config.conv_num_filters_start, self.config)

            rnb = ResNetBlock(cur_num_filters, pri_num_filters, subsample_length, index, self.config)
            layers.append(rnb)

        layers.append(nn.BatchNorm1d(cur_num_filters))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.unsqueeze(x,-1)
        x = x.permute(0,2,1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.rnbs(x)

        x = self.pool(x)
        x = x.view(x.shape[0],-1)
        x = self.fc(x)

        evidence = self.logits2evidence(x)

        return evidence

from tqdm import tqdm
from src.util import evaluate


def ourtrain(model,data_loader,device,optimizer, classes, global_step, annealing_step):
    model.train()

    losses = []
    preds = []
    trues = []
    Us = []

    for i,(data,labels) in enumerate(tqdm(data_loader, desc="train")):

        data = data.to(device)
        labels = labels.to(device)

        evidence = model(data)
        alpha = evidence +1

        loss = ce_loss(labels, alpha, classes, global_step, annealing_step)
        loss = torch.mean(loss)

        S = torch.sum(alpha, dim=1, keepdim=True)
        U = classes / S

        outputs = alpha / S

        losses.append(loss.item())
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0,norm_type=2)
        optimizer.step()
        optimizer.zero_grad()

        preds.extend(outputs.detach().cpu().numpy())
        trues.extend(labels.detach().cpu().numpy())
        Us.extend(U.detach().cpu().numpy().squeeze())

    res = evaluate(trues, preds, desc="train",uncertainty=Us)
    res['loss'] = np.mean(losses)

    return res

def ourtest(model, data_loader, desc,device,classes, global_step, annealing_step):
    losses = []
    preds = []
    trues = []
    Us = []

    model.eval()

    for i, (data, labels) in enumerate(tqdm(data_loader, desc=desc)):
        data = data.to(device)
        labels = labels.to(device)
        evidence = model(data)
        alpha = evidence + 1

        S = torch.sum(alpha, dim=1, keepdim=True)
        U = classes / S
        outputs = alpha / S

        loss = ce_loss(labels, alpha, classes, global_step,annealing_step)
        loss = torch.mean(loss)

        losses.append(loss.item())
        preds.extend(outputs.detach().cpu().numpy())
        trues.extend(labels.detach().cpu().numpy())
        Us.extend(U.detach().cpu().numpy().squeeze())

    res = evaluate(trues, preds, desc=desc,uncertainty=Us)
    res['loss'] = np.mean(losses)

    return res



def run(data_path,device):
    current_file_dir = os.path.basename(__file__)
    log_file = record_logs(current_file_dir)

    print(current_file_dir)

    input_size = 1
    seq_len = Glo.get_value("seq_len")
    num_classes = Glo.get_value("num_classes")

    batch_size = 64
    learning_rate = 3e-4
    num_epochs = 300
    annealing_step = 10

    train_loader, val_loader, test_loader = preparing_data_for_training(data_path,batch_size)

    model = ResNet(input_size, seq_len, num_classes)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0, last_epoch=- 1, verbose=False)

    # SummaryWriter(Tensorboard)
    log_file_dirs = log_file.split('/')
    log_file_dirs = log_file_dirs[0]+'/'+log_file_dirs[1]
    writer = SummaryWriter(f'{log_file_dirs}')

    model.to(device)


    for epoch in range(num_epochs):
        with open(log_file, 'a') as fout:
            print(f'Epoch:{epoch}')
            print(f'Epoch:{epoch}', file=fout)

            training_res = ourtrain(model, train_loader, device, optimizer, num_classes,epoch+1,annealing_step)
            scheduler.step()

            # saveing model
            # torch.save(model, '{0}/model_{1}.pt'.format(log_file_dirs, epoch))
            write_res(training_res, 'train', fout)

            val_res = ourtest(model,val_loader,"val", device, num_classes,global_step=num_epochs,annealing_step=annealing_step)
            write_res(val_res, 'val', fout)

            test_res = ourtest(model, test_loader, "test", device, num_classes, global_step=num_epochs,annealing_step=annealing_step)
            write_res(test_res, 'test', fout)

            write_tensorboard(writer, training_res, val_res, test_res, epoch)

            print('=' * 20, file=fout)

    # save the model for other test
    torch.save(model, '{0}/model_{1}.pt'.format(log_file_dirs, num_epochs))
    writer.close()

