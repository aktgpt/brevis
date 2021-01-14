import torch
import torch.nn as nn

class MaskedMSELoss(torch.nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, input, target, mask):
        diff2 = torch.pow(input-target,2)
        result = torch.sum(diff2) / torch.sum(mask)
        return result