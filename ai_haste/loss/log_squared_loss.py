import torch
import torch.nn as nn
import torch.nn.functional as F


class LogSquaredLoss(nn.Module):
    def __init__(self,):
        super(LogSquaredLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.eps = 1e-8

    def forward(self, output, target):
        # output = torch.log((output) + self.eps)
        # target = torch.log((target) + self.eps)
        # loss = self.l1(target, output)
        loss = torch.log(torch.div(((target) + self.eps), ((output) + self.eps)))
        loss = (loss ** 2).mean()
        return loss

