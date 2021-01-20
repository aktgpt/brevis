import torch
import torch.nn as nn
import torch.nn.functional as F
from .log_squared_loss import LogSquaredLoss
from .median_MAE import MedianMAELoss


def tanh_scaled(x):
    x_tanh = torch.tanh(x)
    x_tanh_scaled = (1 + x_tanh) / 2
    return x_tanh_scaled


class LWMIntLayersL1Loss(nn.Module):
    def __init__(self, pixel_weight=2,activation=tanh_scaled):
        super(LWMIntLayersL1Loss, self).__init__()
        self.l1 = nn.L1Loss(reduction="none")
        self.pixel_weight = pixel_weight
        self.activation = activation

    def forward(self, target, output_mask, target_mask, lwm_op):
        op_label = output_mask.argmax(dim=1).unsqueeze(1)
        weight = torch.eq(op_label, target_mask.unsqueeze(1)).float()
        weight[weight == 0] = self.pixel_weight
        
        lwm_l1_loss = 0.0
        for lwm_layer_op in lwm_op:
            target_resized = F.interpolate(
                target, (lwm_layer_op.shape[2], lwm_layer_op.shape[3])
            )
            weight_resized = F.interpolate(
                weight, (lwm_layer_op.shape[2], lwm_layer_op.shape[3])
            )
            lwm_layer_l1 = (
                self.l1(lwm_layer_op, target_resized) * weight_resized
            ).mean()
            lwm_l1_loss += lwm_layer_l1
        return lwm_l1_loss, weight






