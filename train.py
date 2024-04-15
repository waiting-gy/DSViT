import os
import math
import numpy as np
import random
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from dataloaders.dataloader import MyDataset

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.autograd import Variable
from collections import Counter
from utils.misc import WarmUpLR, AverageMeter
from utils.eval import auc, acc, misClfRes
from tqdm import tqdm
import time
import dataloaders.custom_transforms as tr
from models.dsvit import ViT as vit_3d
import torch.nn.functional as F
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

def setup_seed(seed):
    torch.manual_seed(seed)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
use_cuda = torch.cuda.is_available()
setup_seed(42)

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # read label and data
    label_path = '....'
    patients_train_info = pd.read_excel(label_path, sheet_name=0)
    size = 64
    train_dataset = MyDataset(patient_info=patients_train_info, transform=True, mode='train',
                              size=size)
    val_dataset = MyDataset(patient_info=patients_val_info, transform=False, mode='val',
                            size=size)

    train_dataloader = data.DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_dataloader = data.DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = vit_3d(image_size=64,  # image size
                    frames=64,  # number of frames
                    image_patch_size=4,  # image patch size
                    frame_patch_size=4,  # frame patch size
                    num_classes=2,
                    dim=512,
                    depth=(4, 4, 4, 4),  # (2, 2, 6, 2) (2, 2, 18, 2)
                    heads=(2, 4, 8, 16), 
                    mlp_dim=512,
                    dropout=0.25,
                    emb_dropout=0.25,
                    pool='cls').cuda()  

    model = torch.nn.DataParallel(model).cuda()

    cudnn.benchmark = True
    print('   Model1 Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    label_0 = float(Counter(patients_train_info['data_label'])[0])
    label_1 = float(Counter(patients_train_info['data_label'])[1])
    loss_weight = label_0 / (label_0 + label_1)
    print('loss_weight 0/1: ', loss_weight, 1 - loss_weight)

    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0 - loss_weight, loss_weight]).cuda())
    optimizer = torch.optim.AdamW(model2.parameters(), lr=0.00001,
                                  betas=(0.9, 0.999),
                                  eps=1e-08, weight_decay=0.0005, amsgrad=False)

    log_dir = datetime.now().strftime('model_%Y%m%d_%H%M%S')

    warmup_epoch = 0
    iter_epoch = len(train_dataloader)
    warmup_scheduler2 = WarmUpLR(optimizer2, iter_epoch * warmup_epoch)

    for epoch in range(0, 200):
        warmup_scheduler2 = CosineAnnealingLR(optimizer2, T_max=t_max * iter_epoch, eta_min=0.00001)
        optimizer = optimizer2
        warmup_scheduler = warmup_scheduler2
        state_lr = optimizer.param_groups[0]['lr']
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, 400, state_lr))

        train_loss, train_auc, train_acc, train_data, trainlabel, \
        train_data2, train_data3, train_data4 = train(train_dataloader,
                                                                         model,
                                                                         criterion,
                                                                         optimizer,
                                                                         epoch,
                                                                         use_cuda,
                                                                         warmup_scheduler,
                                                                         is_mixup=False)

        test_loss, test_auc, test_acc, test_data, testlabel, all_loss, all_patient, \
        test_data2, test_data3, test_data4= test(val_dataloader, model, criterion, epoch, use_cuda)


def train(trainloader, model, criterion, optimizer, epoch, use_cuda, scheduler, is_mixup=False):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    auces = AverageMeter()
    acces = AverageMeter()

    alloutput = []
    alltarget = []
    lrs = AverageMeter()
    end = time.time()

    trainloader = tqdm(trainloader)
    for batch_idx, (inputs,  k1, k2, k3, k4, k5, k6, k7, k8, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)
        targets = targets if len(targets)>0 else None
        if not type(inputs) in (tuple, list):
            inputs = inputs
        if use_cuda:
            inputs, k1, k2, k3, k4, k5, k6, k7, k8, targets = inputs.cuda(), k1.cuda(), k2.cuda(), k3.cuda(), k4.cuda(), k5.cuda(), k6.cuda(), k7.cuda(), k8.cuda(), targets.cuda()

        inputs_2 = inputs[:, 1:2, :, :, :]
        inputs_3 = inputs[:, 2:3, :, :, :]
        inputs_4 = inputs[:, 3:4, :, :, :]

        if is_mixup == True:
            inputs = torch.concat([inputs_2, inputs_3, inputs_4], dim=1)
            inputs, targets_a, targets_b, lam = tr.mix_updata(inputs, targets, alpha=1.0)
            inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))
            outputs = model(inputs)
            loss = tr.mixup_criterion(criterion[0], outputs, targets_a, targets_b, lam)
            prec1 = (auc(outputs.data, targets_a.data) + auc(outputs.data, targets_b.data))/2
            acc_pre = (acc(outputs.data, targets_a.data) + acc(outputs.data, targets_b.data))/2
        else:
            inputs = torch.concat([inputs_2, inputs_3, inputs_4], dim=1)
            outputs = model(inputs, k1, k2, k3, k4, k5, k6, k7, k8)
            loss = criterion(outputs, targets) \
                   # + w[0]*criterion(o1, targets) \
                   # + w[1]*criterion(o2, targets) \
                   # + w[2]*criterion(o3, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        alloutput.append(outputs.data)
        alltarget.append(targets.data)
        losses.update(loss.data.item(), targets.size(0))
        auces.update(prec1, targets.size(0))
        acces.update(acc_pre, targets.size(0))
        lrs.update(scheduler.get_last_lr()[0], 1)
      
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        trainloader.set_description('Training Processing ')
        trainloader.set_postfix(train_loss=losses.avg, train_auc=auces.avg,
                               train_acc=acces.avg, lrs=lrs.val)
    alloutputs = torch.cat(alloutput, dim=0)
    alloutputs_f = alloutputs[:,1]
    alltargets = torch.cat(alltarget, dim=0)
    return losses.avg, auc(alloutputs, alltargets), acc(alloutputs, alltargets), alloutputs_f, alltargets, \
           alloutputs2[:,1], alloutputs3[:,1], alloutputs4[:,1]

def test(testloader, model, criterion, epoch, use_cuda, evaluate=False, misClfResPath=None):
    global best_auc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    auces = AverageMeter()
    acces = AverageMeter()
    alloutput = []
    alltarget = []
    all_loss = []
    all_patient = []
    # switch to evaluate mode

    # model1.eval()
    model.eval()

    end = time.time()
    testloader = tqdm(testloader)
    with torch.no_grad():
        for batch_idx, (inputs, k1, k2, k3, k4, k5, k6, k7, k8, targets, patient_name) in enumerate(testloader):
            # measure data loading time
            data_time.update(time.time() - end)

            if not type(inputs) in (tuple, list):
                inputs = inputs
            if use_cuda:
                inputs, k1, k2, k3, k4, k5, k6, k7, k8, targets = inputs.cuda(), k1.cuda(), k2.cuda(), k3.cuda(), k4.cuda(), k5.cuda(), k6.cuda(), k7.cuda(), k8.cuda(), targets.cuda()


            inputs_2 = inputs[:, 1:2, :, :, :]
            inputs_3 = inputs[:, 2:3, :, :, :]
            inputs_4 = inputs[:, 3:4, :, :, :]

            outputs = model(inputs, k1, k2, k3, k4, k5, k6, k7, k8)
            outputs = model(inputs)
            loss = criterion(outputs, targets) \
                   # + w[0] * criterion(o1, targets) \
                   # + w[1] * criterion(o2, targets) \
                   # + w[2] * criterion(o3, targets)

            # measure accuracy and record loss
            alloutput.append(outputs.data)
            alltarget.append(targets.data)
            all_loss.append(loss.data.item())
            all_patient.append(patient_name.data.item())

            prec1 = auc(outputs.data, targets.data)
            acc_pre = acc(outputs.data, targets.data)
            losses.update(loss.data.item(), targets.size(0))
            auces.update(prec1, targets.size(0))
            acces.update(acc_pre, targets.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            testloader.set_description('Validation Processing ')
            testloader.set_postfix(test_loss=losses.avg, test_auc=auces.avg,
                                    test_acc=acces.avg)

        alloutputs = torch.cat(alloutput, dim=0)
        alloutputs_f = alloutputs[:, 1]
        alltargets = torch.cat(alltarget, dim=0)
    return losses.avg, auc(alloutputs, alltargets), acc(alloutputs, alltargets), alloutputs_f, alltargets, \
           all_loss, all_patient, alloutputs2[:,1], alloutputs3[:,1], alloutputs4[:,1]


if __name__ == "__main__":
    main()
