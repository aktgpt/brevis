import torch
import torch.nn as nn
import torch.nn.functional as F


class MedianMAELoss(nn.Module):
    def __init__(self):
        super(MedianMAELoss, self,).__init__()
        self.l1 = nn.L1Loss()

    def forward(self, output, target):
        return self.l1(output, target) / torch.median(target)
