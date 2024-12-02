import torch
import torch.nn as nn
import torch.nn.functional as F


class KLDivergenceLoss(nn.Module):
    """KL Divergence loss implementation"""
    def __init__(self):
        super(KLDivergenceLoss, self).__init__()


    def forward(self, input, target, pad_mask=None, reduction="sum"):
        """
        input: [N, C]
        target: [N, ]
        """
        input_target = F.kl_div(F.log_softmax(input, dim=-1), F.softmax(target, dim=-1), reduction='none')
        target_input = F.kl_div(F.log_softmax(target, dim=-1), F.softmax(input, dim=-1), reduction='none')
        if pad_mask is not None:
            input_target.masked_fill_(pad_mask, 0.)
            target_input.masked_fill_(pad_mask, 0.)
        
        loss = (input_target + target_input) * 0.5

        return loss.mean() if reduction == "mean" else loss.sum()  


class MultiLabelCrossEntropyKLD(nn.Module):
    '''Multi-label Cross Entropy KL Divergence implementation'''
    def __init__(self):
        super(MultiLabelCrossEntropyKLD, self).__init__()


    def forward(self, input, target, reduction="sum"):
        activation1 = torch.sigmoid(input)
        activation2 = torch.sigmoid(target)
        loss = torch.mul(activation2 - activation1, target - input) * 0.5
        
        return loss.mean() if reduction == "mean" else loss.sum()

