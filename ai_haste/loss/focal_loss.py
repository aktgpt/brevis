import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss2d(nn.modules.loss._WeightedLoss):
    def __init__(
        self,
        gamma=2,
        weight=[1, 2],
        size_average=None,
        ignore_index=-100,
        reduce=None,
        reduction="mean",
        balance_param=1.0,
    ):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.pos_weight = torch.tensor(weight).long().reshape(1, 2, 1, 1).cuda()
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.balance_param = balance_param

    def forward(self, input, target): 
        # inputs and targets are assumed to be BatchxClasses
        assert len(input.shape) == len(target.shape)
        assert input.size(0) == target.size(0)
        assert input.size(1) == target.size(1)
        # compute the negative likelyhood
        logpt = -F.binary_cross_entropy_with_logits(
            input,
            target,
            reduction=self.reduction,
            pos_weight=self.pos_weight,
        )
        pt = torch.exp(logpt)
        # compute the loss
        focal_loss = -((1 - pt) ** self.gamma) * logpt
        balanced_focal_loss = self.balance_param * focal_loss
        return balanced_focal_loss
