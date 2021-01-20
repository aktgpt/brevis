import torch
import torch.nn as nn
import torch.nn.functional as F
​
​
class GradientLoss(nn.Module):
    def __init__(
        self, loss=nn.L1Loss(), full=False, alpha=0.1, only_y=False, gms=False,
    ):
        super(GradientLoss, self).__init__()
        self.loss = loss
        self.full = full
        self.alpha = alpha
        self.only_y = only_y
        self.gms = gms
        self.sigma = 1e-5
​
    def gradient(self, x):
        # idea from tf.image.image_gradients(image)
        # https://github.com/tensorflow/tensorflow/blob/r2.1/tensorflow/python/ops/image_ops_impl.py#L3441-L3512
        # x: (b,c,h,w), float32 or float64
        # dx, dy: (b,c,h,w)
​
        h_x = x.size()[-2]
        w_x = x.size()[-1]
        # gradient step=1
        left = x
        right = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
        top = x
        bottom = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]
​
        dx, dy = torch.abs(right - left), torch.abs(bottom - top)
        # dx, dy = right - left, bottom - top
        # dx will always have zeros in the last column, right-left
        # dy will always have zeros in the last row,    bottom-top
        dx[:, :, :, -1] = 0
        dy[:, :, -1, :] = 0
​
        if self.only_y:
            return dy
        else:
            return torch.stack([dx, dy], dim=0)
​
    def forward(self, out_images, target_images):
        d_out_image = self.gradient(out_images)
        d_targe_image = self.gradient(target_images)
​
        if self.gms:
            out_gms = (2 * d_targe_image * d_out_image + self.sigma) / (
                torch.pow(d_targe_image, 2) + torch.pow(d_out_image, 2) + self.sigma
            )
​
            if self.full:
                out_loss = self.alpha * self.loss(out_images, target_images) + torch.std(out_gms)
​
            else:
                out_loss = 1 - torch.mean(out_gms)
​
        else:
​
            if self.full:
                out_loss = self.alpha * self.loss(out_images, target_images) + self.loss(
                    d_out_image, d_targe_image
                )
            else:
                out_loss = self.loss(d_out_image, d_targe_image)
        return out_loss