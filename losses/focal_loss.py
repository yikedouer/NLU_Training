import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiClassFocalLoss(nn.Module):
    """Multi-class Focal loss implementation"""
    def __init__(self, gamma=2, weight=None, ignore_index=-100):
        super(MultiClassFocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        # -(1-p)**γ * log(p); logpt对应log(p)
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1-pt)**self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight, ignore_index=self.ignore_index)
        return loss


class MultiLabelFocalLoss(nn.Module):
    '''Multi-label Focal loss implementation'''
    def __init__(self, alpha=0.25, gamma=2, weight=None, reduction="mean"):
        super(MultiLabelFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.sigmod = nn.Sigmoid()
        self.reduction = reduction

    def forward(self, input, target):
        p = self.sigmod(input)
        zeros = torch.zeros_like(p)
        pos_p_sub = torch.where(target > zeros, target - p, zeros)
        neg_p_sub = torch.where(target > zeros, zeros, p)
        loss = -self.alpha * (pos_p_sub ** self.gamma) * torch.log(p)-(1-self.alpha)*(neg_p_sub ** self.gamma)*torch.log(1.0-p)
        if "mean" == self.reduction:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
