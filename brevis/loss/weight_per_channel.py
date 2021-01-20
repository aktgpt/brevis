import torch
from torch import nn
from .ssim import SSIM


class WeightPerChannel(nn.Module):
    def __init__(self, channel_weights=[10, 1, 5]):
        super(WeightPerChannel, self).__init__()
        self.channel_weights = channel_weights
        self.smoothl1 = nn.SmoothL1Loss(beta=1.0)
        self.ssim = SSIM()

    def forward(self, output, target):
        loss = 0.0
        ssim_loss = 0.0
        l1_loss = 0.0
        for i in range(target.shape[1]):
            channel_l1, channel_ssim = self.calc_loss(output[:, i, :, :], target[:, i, :, :])
            l1_loss += channel_l1
            ssim_loss += channel_ssim
            loss += self.channel_weights[i] * (channel_l1 + 0.1 * channel_ssim)
            # loss += self.channel_weights[i] * (
            #     self.smoothl1()
            #     + 0.1 * self.ssim(output[:, i, :, :].unsqueeze(0), target[:, i, :, :].unsqueeze(0))
            # )
        return loss, l1_loss / target.shape[1], ssim_loss / target.shape[1]

    def calc_loss(self, output, target):
        loss_l1 = self.smoothl1(output, target)
        loss_ssim = self.ssim(output.unsqueeze(0), target.unsqueeze(0))
        return loss_l1, loss_ssim
