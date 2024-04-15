import os
import sys
import time
import math
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.optim.lr_scheduler import _LRScheduler
from collections import defaultdict
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, confusion_matrix

def acc(output, target):
    """compute the metrics without threshold"""
    # batch_size = target.size(0)
    pre_pos = output.cpu().numpy()
    res = pre_pos[:, 0] < pre_pos[:, 1]
    res = res.astype(int)
    true = target
    result = accuracy_score(true, res)
    tn, fp, fn, tp = confusion_matrix(true, res).ravel()
    sen = tp/(tp + fn)
    spe = tn/(tn + fp)
    return result, sen, spe

from __future__ import print_function, absolute_import
from sklearn.metrics import roc_auc_score, accuracy_score
import torch
import numpy as np
import pandas as pd
__all__ = ['accuracy', 'auc', 'acc', 'misClfRes']

def softmax(X):
    exps = np.exp(X)
    return exps / np.sum(exps, axis=0)

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def auc(output, target):
    """Computes the precision@k for the specified values of k"""
    # batch_size = target.size(0)
    pre_pos = output[:, 1].cpu().numpy()
    # pre_pos = output.max(1)[0].cpu().numpy()
    true = target.cpu().numpy() # [:, 1]
    if true.all() == 1 or true.any() == 0:
        return np.nan
    res = roc_auc_score(true, pre_pos)
    # print(pre_pos)
    return res

def misClfRes(output, target, path):
    """print the index of misclassfied sample"""
    pre_two = output.cpu().numpy()
    pre_pro = np.apply_along_axis(softmax, 1, pre_two)
    pre = (pre_two[:, 0] < pre_two[:, 1]).astype(int)
    label = target.cpu().numpy()# [:, 1]
    pd.DataFrame({'pro':pre_pro[:, 1], 'pre': pre, 'label':label, 'clf':pre==label}).to_csv(path, index=False)

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=True, reduction='elementwise_mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.logits:
            # BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
            BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction is None:
            return F_loss
        else:
            return torch.mean(F_loss)

class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_prods = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_prods).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = -(targets * log_prods).mean(0).sum()
        return loss


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimizer(e.g. sgd)
        total_iters: total_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> float:
        """we will use the first m batches, and set the training rate to base_lr * m / total_iters
        """
        return [0.000 + base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]
        # mae
        # return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]
