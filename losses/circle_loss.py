import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLabelCircleLoss(nn.Module):
    def __init__(self, reduction="mean", inf=1e12):
        """Circle Loss of Multi Label
        多标签分类的交叉熵
        quota: [将“softmax+交叉熵”推广到多标签分类问题](https://spaces.ac.cn/archives/7359)
        args:
            reduction: str, Specifies the reduction to apply to the output, 输出形式. 
                            eg.``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``
            inf: float, Minimum of maths, 无穷大.  eg. 1e12
        returns:
            Tensor of loss.
        examples:
            >>> label, logits = [[1, 1, 1, 1], [0, 0, 0, 1]], [[0, 1, 1, 0], [1, 0, 0, 1],]
            >>> label, logits = torch.tensor(label).float(), torch.tensor(logits).float()
            >>> loss = MultiLabelCircleLoss()(logits, label)
        """
        super(MultiLabelCircleLoss, self).__init__()
        self.reduction = reduction
        self.inf = inf

    def forward(self, logits, labels):
        batch_size, label_size = logits.shape[:2]
        labels = labels.reshape(batch_size * label_size, -1)
        logits = logits.reshape(batch_size * label_size, -1)

        logits = (1 - 2 * labels) * logits
        logits_neg = logits - labels * self.inf
        logits_pos = logits - (1 - labels) * self.inf
        zeros = torch.zeros_like(logits[..., :1])
        logits_neg = torch.cat([logits_neg, zeros], dim=-1)
        logits_pos = torch.cat([logits_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(logits_neg, dim=-1)
        pos_loss = torch.logsumexp(logits_pos, dim=-1)
        loss = neg_loss + pos_loss

        return loss.mean() if "mean" == self.reduction else loss.sum()

