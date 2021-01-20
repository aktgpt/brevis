# Python Standard Library imports
import types

# Other imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

# Local imports
from .model import BaseModel
from .layers.BlurPool2d import BlurPool2d


def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        if norm is not None:
            concated_features = norm(concated_features)
        bottleneck_output = conv(relu(concated_features))
        return bottleneck_output

    return bn_function


def tanh_scaled(x):
    x_tanh = torch.tanh(x)
    x_tanh_scaled = (1 + x_tanh) / 2
    return x_tanh_scaled


def icnr_init(x, scale=2, init=nn.init.kaiming_normal_):
    "ICNR init of `x`, with `scale` and `init` function"
    ni, nf, h, w = x.shape
    ni2 = int(ni / (scale ** 2))
    k = init(x.new_zeros([ni2, nf, h, w])).transpose(0, 1)
    k = k.contiguous().view(ni2, nf, -1)
    k = k.repeat(1, 1, scale ** 2)
    return k.contiguous().view([nf, ni, h, w]).transpose(0, 1)


class DenseLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        growth_rate,
        batch_norm="batchnorm",
        dropout=0.2,
        efficient=False,
    ):
        super().__init__()
        self.dropout = dropout
        self.efficient = efficient
        self.has_bn = batch_norm is not None
        # Standard Tiramisu Layer (BN-ReLU-Conv-DropOut)
        if batch_norm == "batchnorm":
            self.add_module("batchnorm", nn.BatchNorm2d(in_channels))
        self.add_module("relu", nn.ReLU(True))
        self.add_module(
            "conv",
            nn.Conv2d(
                in_channels, growth_rate, kernel_size=3, stride=1, padding=1, bias=True
            ),
        )

    def any_requires_grad(self, x):
        """Returns True if any of the layers in x requires gradients.
        """
        return any(layer.requires_grad for layer in x)

    def forward(self, x):
        bn_function = _bn_function_factory(self.batchnorm, self.relu, self.conv)
        if self.efficient and self.any_requires_grad(x):
            x = cp.checkpoint(bn_function, *x)
        else:
            x = bn_function(*x)
        if self.dropout and self.dropout != 0.0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class DenseBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        growth_rate,
        nb_layers,
        upsample=False,
        batch_norm=True,
        dropout=0.2,
        efficient=False,
    ):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList(
            [
                DenseLayer(
                    in_channels + i * growth_rate,
                    growth_rate,
                    batch_norm,
                    dropout,
                    efficient,
                )
                for i in range(nb_layers)
            ]
        )
        self.upsample = upsample

    def forward(self, x):
        skip_connections = [x]

        for layer in self.layers:
            out = layer(skip_connections)
            skip_connections.append(out)

        if self.upsample:
            # Returns all of the x's, except for the first x (input), concatenated
            # As we are not supposed to have skip connections over the full dense block.
            # See original tiramisu paper for more details
            return torch.cat(skip_connections[1:], 1)
        else:
            # Returns all of the x's concatenated
            return torch.cat(skip_connections, 1)


class TransitionDown(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        batch_norm="batchnorm",
        dropout=0.2,
        pooling="max",
    ):
        super().__init__()
        if batch_norm == "batchnorm":
            self.add_module("batchnorm", nn.BatchNorm2d(in_channels))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module(
            "conv",
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
        )
        if dropout and dropout != 0.0:
            self.add_module("dropout", nn.Dropout2d(0.2))
        if pooling == "max":
            self.add_module("pool", nn.MaxPool2d(2))
        elif pooling == "avg":
            self.add_module("pool", nn.AvgPool2d(2))
        elif pooling == "blurpool":
            self.add_module("blurpool", BlurPool2d(out_channels))

    def forward(self, x):
        return super().forward(x)


class TransitionUp(nn.Module):
    def __init__(self, in_channels, out_channels, upsampling_type="deconv"):
        super().__init__()
        if upsampling_type == "upsample":
            self.upsampling_layer = nn.Sequential(
                nn.UpsamplingNearest2d(scale_factor=2),
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=True,
                ),
                nn.ReLU(inplace=True),
            )
        elif upsampling_type == "pixelshuffle":
            self.upsampling_layer = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=4 * out_channels,
                    kernel_size=3,
                    padding=True,
                    bias=True,
                ),
                nn.ReLU(inplace=True),
                nn.PixelShuffle(2),
            )
        elif upsampling_type == "pixelshuffle_blur":
            self.upsampling_layer = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=4 * out_channels,
                    kernel_size=3,
                    padding=True,
                    bias=True,
                ),
                nn.ReLU(inplace=True),
                nn.PixelShuffle(2),
                nn.ReplicationPad2d((1, 0, 1, 0)),
                nn.AvgPool2d(2, stride=1),
            )
            self.upsampling_layer[0].weight.data.copy_(
                icnr_init(self.upsampling_layer[0].weight.data)
            )

        else:  # Default: "deconv"
            self.upsampling_layer = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=0,
                    bias=True,
                ),
                nn.ReLU(inplace=True),
            )

    def forward(self, x, skip, use_skip=False):
        if use_skip:
            out = self.upsampling_layer(x)
            out = center_crop(out, skip.size(2), skip.size(3))
            out = torch.cat([skip, out], 1)
        else:
            out = self.upsampling_layer(x)
        return out


def center_crop(layer, max_height, max_width):
    """ Crops a given to a certain size by removing equal margins all around."""
    _, _, h, w = layer.size()
    xy1 = (w - max_width) // 2
    xy2 = (h - max_height) // 2
    return layer[:, :, xy2 : (xy2 + max_height), xy1 : (xy1 + max_width)]


class DenseUNetLUPISkipSum(BaseModel):
    """DensUNet
    Paper: The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for
    Semantic Segmentation
    URL: https://arxiv.org/pdf/1611.09326.pdf
    Notes:
        Coded with the help of https://github.com/bfortuner/pytorch_tiramisu
        MIT License - Copyright (c) 2018 Brendan Fortuner
        and the help of https://github.com/keras-team/keras-contrib
        MIT License - Copyright (c) 2017 Fariz Rahman
    """

    def __init__(
        self,
        in_channels=3,
        nb_classes=1,
        nb_seg_classes=2,
        init_conv_size=3,
        init_conv_filters=48,
        init_conv_stride=1,
        down_blocks=(4, 4, 4, 4, 4),
        bottleneck_layers=4,
        up_blocks=(4, 4, 4, 4, 4),
        growth_rate=12,
        compression=1.0,
        dropout_rate=0.2,
        upsampling_type="upsample",
        early_transition=False,
        transition_pooling="max",
        batch_norm="batchnorm",
        include_top=True,
        activation_func=None,
        efficient=False,
        use_lwm=False,
    ):
        """ Creates a Tiramisu/Fully Convolutional DenseNet Neural Network for
        image segmentation.

        Args:
            nb_classes: The number of classes to predict.
            in_channels: The number of channels of the input images.
            init_conv_size: The size of the very first first layer.
            init_conv_filters: The number of filters of the very first layer.
            init_conv_stride: The stride of the very first layer.
            down_blocks: The number of DenseBlocks and their size in the
                compressive part.
            bottleneck_layers: The number of DenseBlocks and their size in the
                bottleneck part.
            up_blocks: The number of DenseBlocks and their size in the
                reconstructive part.
            growth_rate: The rate at which the DenseBlocks layers grow.
            compression: Optimization where each of the DenseBlocks layers are reduced
                by a factor between 0 and 1. (1.0 does not change the original arch.)
            dropout_rate: The dropout rate to use.
            upsampling_type: The type of upsampling to use in the TransitionUp layers.
                available options: ["upsample" (default), "deconv", "pixelshuffle"]
                For Pixel shuffle see: https://arxiv.org/abs/1609.05158
            early_transition: Optimization where the input is downscaled by a factor
                of two after the first layer. You can thus reduce the numbers of down
                and up blocks by 1.
            transition_pooling: The type of pooling to use during the transitions.
                available options: ["max" (default), "avg", "blurpool"]
            batch_norm: Type of batch normalization to use.
                available options: ["batchnorm" (default), None]
                For FRN see: https://arxiv.org/pdf/1911.09737v1.pdf
            include_top (bool): Including the top layer, with the last convolution
                and softmax/sigmoid (True) or returns the embeddings for each pixel
                of the input image (False).
            activation_func (func): Activation function to use at the end of the model.
            efficient (bool): Memory efficient version of the Tiramisu.
                See: https://arxiv.org/pdf/1707.06990.pdf
        """
        super().__init__(in_channels, nb_classes, activation_func)
        self.nb_classes = nb_classes
        self.nb_seg_classes = nb_seg_classes
        self.init_conv_filters = init_conv_filters
        self.down_blocks = down_blocks
        self.bottleneck_layers = bottleneck_layers
        self.up_blocks = up_blocks
        self.growth_rate = growth_rate
        self.compression = compression
        self.early_transition = early_transition
        self.include_top = include_top
        self.use_lwm = use_lwm

        channels_count = init_conv_filters
        channels_count2 = init_conv_filters
        skip_connections = []

        # Check
        assert upsampling_type in [
            "deconv",
            "upsample",
            "pixelshuffle",
            "pixelshuffle_blur",
        ], "upsampling_type option does not exist."
        assert transition_pooling in [
            "max",
            "avg",
            "blurpool",
        ], "transition_pooling option does not exist."
        assert batch_norm in ["batchnorm", None], "batch_norm option does not exist."

        # First layer
        init_conv_padding = (init_conv_size - 1) >> 1
        self.add_module(
            "conv_init",
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=init_conv_filters,
                kernel_size=init_conv_size,
                stride=init_conv_stride,
                padding=init_conv_padding,
                bias=True,
            ),
        )

        if early_transition:
            self.add_module(
                "early_transition_down",
                TransitionDown(
                    in_channels=channels_count,
                    out_channels=int(channels_count * compression),
                    batch_norm=batch_norm,
                    dropout=dropout_rate,
                    pooling=transition_pooling,
                ),
            )
            channels_count = int(channels_count * compression)

        # Downsampling part
        self.layers_down = nn.ModuleList([])
        self.transitions_down = nn.ModuleList([])
        for block_size in self.down_blocks:
            self.layers_down.append(
                DenseBlock(
                    in_channels=channels_count,
                    growth_rate=growth_rate,
                    nb_layers=block_size,
                    upsample=False,
                    batch_norm=batch_norm,
                    dropout=dropout_rate,
                    efficient=efficient,
                )
            )
            channels_count += growth_rate * block_size
            skip_connections.insert(0, channels_count)
            self.transitions_down.append(
                TransitionDown(
                    in_channels=channels_count,
                    out_channels=int(channels_count * compression),
                    batch_norm=batch_norm,
                    dropout=dropout_rate,
                    pooling=transition_pooling,
                )
            )
            channels_count = int(channels_count * compression)

        # Bottleneck
        self.add_module(
            "bottleneck",
            DenseBlock(
                in_channels=channels_count,
                growth_rate=growth_rate,
                nb_layers=bottleneck_layers,
                upsample=True,
                batch_norm=batch_norm,
                dropout=dropout_rate,
                efficient=efficient,
            ),
        )
        prev_block_channels = growth_rate * bottleneck_layers
        channels_count += prev_block_channels

        prev_block_channels2 = growth_rate * bottleneck_layers
        channels_count2 += prev_block_channels2

        # Upsampling part (First Decoder)
        self.layers_up = nn.ModuleList([])
        self.transitions_up = nn.ModuleList([])
        for i, block_size in enumerate(self.up_blocks[:-1]):
            self.transitions_up.append(
                TransitionUp(
                    in_channels=prev_block_channels,
                    out_channels=prev_block_channels,
                    upsampling_type=upsampling_type,
                )
            )
            channels_count = prev_block_channels
            self.layers_up.append(
                DenseBlock(
                    in_channels=channels_count,
                    growth_rate=growth_rate,
                    nb_layers=block_size,
                    upsample=True,
                    batch_norm=batch_norm,
                    dropout=dropout_rate,
                    efficient=efficient,
                )
            )
            prev_block_channels = growth_rate * block_size
            channels_count += prev_block_channels

        self.transitions_up.append(
            TransitionUp(
                in_channels=prev_block_channels,
                out_channels=prev_block_channels,
                upsampling_type=upsampling_type,
            )
        )
        channels_count = prev_block_channels
        self.layers_up.append(
            DenseBlock(
                in_channels=channels_count,
                growth_rate=growth_rate,
                nb_layers=up_blocks[-1],
                upsample=False,
                batch_norm=batch_norm,
                dropout=dropout_rate,
                efficient=efficient,
            )
        )
        channels_count += growth_rate * up_blocks[-1]

        if early_transition:
            self.add_module(
                "early_transition_up",
                TransitionUp(
                    in_channels=channels_count,
                    out_channels=channels_count,
                    upsampling_type=upsampling_type,
                ),
            )
            channels_count += init_conv_filters

        # Last layer
        if include_top:
            # self.final_conv = nn.Sequential(
            #     DenseBlock(
            #         in_channels=channels_count,
            #         growth_rate=12,
            #         nb_layers=4, 
            #         upsample=False,
            #         batch_norm=batch_norm,
            #         dropout=dropout_rate,
            #         efficient=efficient,
            #     ),
            #     nn.Conv2d(
            #         in_channels=channels_count + 12 * 4,
            #         out_channels=nb_seg_classes,
            #         kernel_size=1,
            #         stride=1,
            #         padding=0,
            #         bias=True,
            #     ),
            # )
            self.final_conv = nn.Conv2d(
                in_channels=channels_count,
                out_channels=nb_seg_classes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )

        # Upsampling part (Second Decoder)
        self.layers_up2 = nn.ModuleList([])
        self.transitions_up2 = nn.ModuleList([])
        if self.use_lwm:
            self.lwm_layers = nn.ModuleList([])
        for i, block_size in enumerate(self.up_blocks[:-1]):
            self.transitions_up2.append(
                TransitionUp(
                    in_channels=prev_block_channels2,
                    out_channels=prev_block_channels2,
                    upsampling_type=upsampling_type,
                )
            ) 
            channels_count2 = prev_block_channels2 + skip_connections[i]
            self.layers_up2.append(
                DenseBlock(
                    in_channels=channels_count2,
                    growth_rate=growth_rate,
                    nb_layers=block_size,
                    upsample=True,
                    batch_norm=batch_norm,
                    dropout=dropout_rate,
                    efficient=efficient,
                )
            )
            prev_block_channels2 = growth_rate * block_size
            channels_count2 += prev_block_channels2
            if self.use_lwm:
                self.lwm_layers.append(
                nn.Conv2d(
                    in_channels=prev_block_channels2,
                    out_channels=nb_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True,
                )
            )

        self.transitions_up2.append(
            TransitionUp(
                in_channels=prev_block_channels2,
                out_channels=prev_block_channels2,
                upsampling_type=upsampling_type,
            )
        )
        channels_count2 = prev_block_channels2 + skip_connections[-1]
        self.layers_up2.append(
            DenseBlock(
                in_channels=channels_count2,
                growth_rate=growth_rate,
                nb_layers=up_blocks[-1],
                upsample=False,
                batch_norm=batch_norm,
                dropout=dropout_rate,
                efficient=efficient,
            )
        )

        channels_count2 += growth_rate * up_blocks[-1]

        if early_transition:
            self.add_module(
                "early_transition_up", 
                TransitionUp(
                    in_channels=channels_count2,
                    out_channels=channels_count2,
                    upsampling_type=upsampling_type,
                ),
            )
            channels_count2 += init_conv_filters

        # Last layer
        if include_top:
            # self.final_conv2 = nn.Conv2d(
            #     in_channels=channels_count2,
            #     out_channels=nb_classes,
            #     kernel_size=1,
            #     stride=1,
            #     padding=0,
            #     bias=True,
            # )
            self.final_conv2 = nn.Sequential(
                DenseBlock(
                    in_channels=channels_count2,
                    growth_rate=12,
                    nb_layers=4, 
                    upsample=False,
                    batch_norm=batch_norm,
                    dropout=dropout_rate,
                    efficient=efficient,
                ),
                nn.Conv2d(
                    in_channels=channels_count2 + 12 * 4,
                    out_channels=nb_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True,
                ),
            )
            self.final_activation2 = self.activation_func

    def forward(self, x):
        x = self.conv_init(x)

        transition_skip = None 
        if self.early_transition:
            transition_skip = x
            x = self.early_transition_down(x)

        skip_connections = []
        for i in range(len(self.down_blocks)):
            x = self.layers_down[i](x)
            skip_connections.append(x)
            x = self.transitions_down[i](x)

        x = self.bottleneck(x)

        ## First Decoder
        decoder_sum_connections = []
        for i in range(0, len(self.up_blocks)):
            if i == 0:
                x1 = self.transitions_up[i](x, None)
            else:
                decoder_sum_connections.append(x1)
                x1 = self.transitions_up[i](x1, None)
            x1 = self.layers_up[i](x1)

        if self.early_transition:
            x1 = self.early_transition_up(x1, skip=transition_skip)
        if self.include_top:
            # Computation of the final 1x1 convolution
            x1 = self.final_conv(x1)

        ## Second Decoder
        if self.use_lwm:
            lwm_ops = []
        for i in range(0, len(self.up_blocks)):
            skip = skip_connections.pop()
            if i == 0:
                x2 = self.transitions_up2[i](x, skip, use_skip=True)
            else:
                sum_layers = decoder_sum_connections.pop(0)
                x2 = self.transitions_up2[i](x2 + sum_layers, skip, use_skip=True)
            x2 = self.layers_up2[i](x2)
            if self.training and self.use_lwm:
                if i < len(self.lwm_layers):
                    lwm_ops.append(self.lwm_layers[i](x2))

        if self.early_transition:
            x2 = self.early_transition_up2(x2, skip=transition_skip)

        if self.include_top:
            # Computation of the final 1x1 convolution
            x2 = self.final_conv2(x2)
            if self.final_activation2:
                x2 = self.final_activation2(x2)

        if self.training and self.use_lwm:
            return x1, x2, lwm_ops
        else:
            return x1, x2
