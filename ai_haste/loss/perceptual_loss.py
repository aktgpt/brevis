import torch
from torch import nn
from torchvision import models, transforms


def contentFunc():
    conv_3_3_layer = 14
    cnn = models.vgg19(pretrained=True).features
    cnn = cnn.cuda()
    model = nn.Sequential()
    model = model.cuda()
    model = model.eval()
    for i, layer in enumerate(list(cnn)):
        model.add_module(str(i), layer)
        if i == conv_3_3_layer:
            break
    return model


class PerceptualLoss:
    def initialize(self, loss):
        with torch.no_grad():
            self.criterion = loss
            self.contentFunc = contentFunc()
            self.transform = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )

    def get_loss(self, fakeIm, realIm):
        fakeIm = fakeIm.repeat(1, 3, 1, 1) / 255
        realIm = realIm.repeat(1, 3, 1, 1) / 255
        fakeIm = (fakeIm + 1) / 2.0
        realIm = (realIm + 1) / 2.0
        fakeIm[0, :, :, :] = self.transform(fakeIm[0, :, :, :])
        realIm[0, :, :, :] = self.transform(realIm[0, :, :, :])
        f_fake = self.contentFunc.forward(fakeIm)
        f_real = self.contentFunc.forward(realIm)

        f_real_no_grad = f_real.detach()
        loss = self.criterion(f_fake, f_real_no_grad)
        return 0.5 * torch.mean(loss) + 0.5 * nn.MSELoss()(fakeIm, realIm)

    def __call__(self, fakeIm, realIm):
        return self.get_loss(fakeIm, realIm)
