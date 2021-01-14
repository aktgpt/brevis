import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedBCELogitsLoss(nn.Module):
    def __init__(self, weights=[1, 1.5]):
        super(WeightedBCELogitsLoss, self,).__init__()
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(weights).reshape(1, 2, 1, 1).cuda()
        )

    def forward(self, output, target):
        return self.bce(output, target)
