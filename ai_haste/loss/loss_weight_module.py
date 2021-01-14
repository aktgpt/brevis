import torch
import torch.nn as nn
import torch.nn.functional as F
from .log_squared_loss import LogSquaredLoss
from .median_MAE import MedianMAELoss


def tanh_scaled(x):
    x_tanh = torch.tanh(x)
    x_tanh_scaled = (1 + x_tanh) / 2
    return x_tanh_scaled


class LWMIntLayersBCEL1Loss(nn.Module):
    def __init__(self, pixel_weight=2,activation=tanh_scaled):
        super(LWMIntLayersBCEL1Loss, self).__init__()
        self.l1 = nn.L1Loss(reduction="none")
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.pixel_weight = pixel_weight
        self.activation = activation

    def forward(self, target, output_mask, target_mask, lwm_op):
        op_label = output_mask.argmax(dim=1).unsqueeze(1)
        bce_weight = torch.eq(op_label, target_mask.unsqueeze(1)).float()
        # bce_weight[bce_weight == 0] = self.pixel_weight
        target_onehot = (
            F.one_hot(target_mask.unsqueeze(1).long()).squeeze(1).permute(0, 3, 1, 2)
        ).float()

        bce_loss = self.bce(output_mask, target_onehot)
        weight = bce_weight.clone()
        weight[torch.where(bce_weight == 0)] = 1 + bce_loss[torch.where(bce_weight == 0)]
        # weight = torch.eq(op_label, target_mask.unsqueeze(1)).float()
        # weight[weight == 0] = self.pixel_weight
        
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

class LWMIntLayersMedianMAEBCELoss(nn.Module):
    def __init__(
        self, pixel_weight=2, loss_weight=0.5, lwm_weight=0.1, activation=tanh_scaled
    ):
        super(LWMIntLayersMedianMAEBCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.l1 = nn.L1Loss(reduction="none")
        self.pixel_weight = pixel_weight
        self.loss_weight = loss_weight
        self.activation = activation
        self.lwm_weight = lwm_weight
        self.eps = 1e-8

    def forward(self, output, target, output_mask, target_mask, lwm_op):
        op_label = output_mask.argmax(dim=1).unsqueeze(1)
        bce_weight = torch.eq(op_label, target_mask.unsqueeze(1)).float()
        bce_weight[bce_weight == 0] = self.pixel_weight
        target_onehot = (
            F.one_hot(target_mask.unsqueeze(1).long()).squeeze(1).permute(0, 3, 1, 2)
        ).float()

        bce_loss = self.bce(output_mask, target_onehot)
        weighted_bce = torch.mean(bce_loss * bce_weight)

        weight = bce_weight.clone()
        weight[torch.where(bce_weight == self.pixel_weight)] = (
            1 + bce_loss[torch.where(bce_weight == self.pixel_weight)]
        )

        median_mae = self.l1(output, target) / torch.median(target)
        weighted_mae = torch.mean(median_mae * weight)

        lwm_mae_loss = 0.0
        for lwm_layer_op in lwm_op:
            target_resized = F.interpolate(
                target, (lwm_layer_op.shape[2], lwm_layer_op.shape[3])
            )
            weight_resized = F.interpolate(
                weight, (lwm_layer_op.shape[2], lwm_layer_op.shape[3])
            )
            layer_mae_loss = self.l1(
                self.activation(lwm_layer_op), target_resized
            ) / torch.median(target_resized)
            lwm_mae_loss += (layer_mae_loss * weight_resized).mean()
        return (
            (self.loss_weight * weighted_mae)
            + weighted_bce
            + self.lwm_weight * lwm_mae_loss
        )


class LWMIntLayersMedianMAELoss(nn.Module):
    def __init__(
        self, pixel_weight=2, loss_weight=0.5, lwm_weight=0.1, activation=tanh_scaled
    ):
        super(LWMIntLayersMedianMAELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.l1 = nn.L1Loss(reduction="none")
        self.pixel_weight = pixel_weight
        self.loss_weight = loss_weight
        self.activation = activation
        self.lwm_weight = lwm_weight
        self.eps = 1e-8

    def forward(self, output, target, output_mask, target_mask, lwm_op):
        op_label = output_mask.argmax(dim=1).unsqueeze(1)
        weight = torch.eq(op_label, target_mask.unsqueeze(1)).float()
        weight[weight == 0] = self.pixel_weight
        target_onehot = (
            F.one_hot(target_mask.unsqueeze(1).long()).squeeze(1).permute(0, 3, 1, 2)
        ).float()

        bce_loss = self.bce(output_mask, target_onehot)
        weighted_bce = torch.mean(bce_loss * weight)

        median_mae = self.l1(output, target) / torch.median(target)
        weighted_mae = torch.mean(median_mae * weight)

        lwm_mae_loss = 0.0
        for lwm_layer_op in lwm_op:
            target_resized = F.interpolate(
                target, (lwm_layer_op.shape[2], lwm_layer_op.shape[3])
            )
            weight_resized = F.interpolate(
                weight, (lwm_layer_op.shape[2], lwm_layer_op.shape[3])
            )
            layer_mae_loss = self.l1(
                self.activation(lwm_layer_op), target_resized
            ) / torch.median(target_resized)
            lwm_mae_loss += (layer_mae_loss * weight_resized).mean()
        return (
            (self.loss_weight * weighted_mae)
            + weighted_bce
            + self.lwm_weight * lwm_mae_loss
        )





class LWMIntLayersLogSqrdLoss(nn.Module):
    def __init__(
        self, pixel_weight=2, loss_weight=0.5, lwm_weights=0.1, activation=tanh_scaled
    ):
        super(LWMIntLayersLogSqrdLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.pixel_weight = pixel_weight
        self.loss_weight = loss_weight
        self.activation = activation
        self.lwm_weights = lwm_weights
        self.eps = 1e-8

    def forward(self, output, target, output_mask, target_mask, lwm_op):
        op_label = output_mask.argmax(dim=1).unsqueeze(1)
        weight = torch.eq(op_label, target_mask.unsqueeze(1)).float()
        weight[weight == 0] = self.pixel_weight
        target_onehot = (
            F.one_hot(target_mask.unsqueeze(1).long()).squeeze(1).permute(0, 3, 1, 2)
        ).float()

        bce_loss = self.bce(output_mask, target_onehot)
        weighted_bce = torch.mean(bce_loss * weight)

        log_loss = (
            torch.log(torch.div(((target) + self.eps), ((output) + self.eps)))
        ) ** 2
        # loss = (loss ** 2).mean()

        # l1_loss = self.l1(output, target)
        weighted_log_loss = torch.mean(log_loss * weight)

        lwm_log_loss = 0.0
        for lwm_layer_op in lwm_op:
            target_resized = F.interpolate(
                target, (lwm_layer_op.shape[2], lwm_layer_op.shape[3])
            )
            weight_resized = F.interpolate(
                weight, (lwm_layer_op.shape[2], lwm_layer_op.shape[3])
            )
            lwm_layer_loss = (
                torch.log(
                    torch.div(
                        ((target_resized) + self.eps),
                        ((self.activation(lwm_layer_op)) + self.eps),
                    )
                )
                ** 2
            )
            lwm_log_loss += (lwm_layer_loss * weight_resized).mean()
        return (
            (self.loss_weight * weighted_log_loss)
            + weighted_bce
            + self.lwm_weights * lwm_log_loss
        )


class LWMLoss(nn.Module):
    def __init__(self, pixel_weight=2):
        super(LWMLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.l1 = nn.L1Loss(reduction="none")
        self.pixel_weight = pixel_weight

    def forward(self, output, target, output_mask, target_mask):
        op_label = output_mask.argmax(dim=1).unsqueeze(1)
        weight = torch.eq(op_label, target_mask.unsqueeze(1)).float()
        weight[weight == 0] = self.pixel_weight
        target_onehot = (
            F.one_hot(target_mask.unsqueeze(1).long()).squeeze(1).permute(0, 3, 1, 2)
        ).float()
        bce_loss = self.bce(output_mask, target_onehot)
        weighted_bce = torch.mean(bce_loss * weight)

        l1_loss = self.l1(output, target)
        weighted_l1 = torch.mean(l1_loss * weight)
        return weighted_l1 + weighted_bce


class LWMLogSqrdLoss(nn.Module):
    def __init__(self, pixel_weight=2, loss_weight=0.5):
        super(LWMLogSqrdLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.pixel_weight = pixel_weight
        self.loss_weight = loss_weight
        self.eps = 1e-8

    def forward(self, output, target, output_mask, target_mask):
        op_label = output_mask.argmax(dim=1).unsqueeze(1)
        weight = torch.eq(op_label, target_mask.unsqueeze(1)).float()
        weight[weight == 0] = self.pixel_weight
        target_onehot = (
            F.one_hot(target_mask.unsqueeze(1).long()).squeeze(1).permute(0, 3, 1, 2)
        ).float()

        bce_loss = self.bce(output_mask, target_onehot)
        weighted_bce = torch.mean(bce_loss * weight)

        log_loss = (
            torch.log(torch.div(((target) + self.eps), ((output) + self.eps)))
        ) ** 2
        # loss = (loss ** 2).mean()

        # l1_loss = self.l1(output, target)
        weighted_log_loss = torch.mean(log_loss * weight)
        return (self.loss_weight * weighted_log_loss) + weighted_bce
