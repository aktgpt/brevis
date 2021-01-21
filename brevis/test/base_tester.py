import os
import random

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error


def calculate_ssim(im1, im2, data_range=255, multichannel=True):
    if multichannel:
        full_ssim = ssim(im1, im2, val_range=data_range, multichannel=True, full=True)[1]
        out_ssim = full_ssim.mean()
    else:
        full_ssim = ssim(im1, im2, val_range=data_range, multichannel=False, full=True)[1]
        out_ssim = full_ssim.mean()

    return out_ssim


class BaseTester:
    def __init__(self, config, save_folder):
        self.config = config
        self.save_folder = save_folder
        self.model_checkpoint = torch.load(
            os.path.join(save_folder, config["model_path"])
        )
        self.image_folder = os.path.join(self.save_folder, "test_images")
        if not os.path.exists(self.image_folder):
            os.makedirs(self.image_folder)

    def test(self, dataloader, models):
        # self.models = []
        # for model in models:
        model = models[0].cuda()
        self.model = torch.nn.DataParallel(model)

        # self.model = self.models[0]

        self.model.load_state_dict(self.model_checkpoint["model_state_dict"])

        self.model.eval()

        best_train_epoch = self.model_checkpoint["epoch"]
        best_train_loss = self.model_checkpoint["epoch_loss"]
        print(f"Epoch:{best_train_epoch},Loss:{best_train_loss}")

        test_ssim, test_loss = self._test_epoch(dataloader)

        df = pd.DataFrame({"SSIM": test_ssim, "MAE": test_loss})
        df.to_csv(os.path.join(self.save_folder, "losses_test.csv"), index=False)

    def _test_epoch(self, dataloader):
        MAE_criterion = torch.nn.L1Loss()

        ssims = []
        MAEs = []
        target_ims = []
        
        with torch.no_grad():
            for batch_idx, sample in enumerate(tqdm(dataloader)):
                image_name = sample[2]
                preprocess_step = sample[3][0]
                preprocess_stats = sample[4]

                input = sample[0].cuda().to(non_blocking=True)
                target = sample[1].unsqueeze(1).cuda().to(non_blocking=True)
                C_out = target.shape[1]

                output = self.infer_full_image(input, C_out, kernel_size=512, stride=256)

                self.write_output_images(
                    output[0], image_name, preprocess_step, preprocess_stats
                )

                MAEs.append( 
                    (
                        MAE_criterion(
                            output.float(),
                            target.float(),
                        )  
                    ).item()
                )
                
                target_ims.append(target.detach().cpu().numpy().astype('float32'))
 
                output = (
                    (output[0] * 255)
                    .detach()
                    .cpu()
                    .numpy()
                    .transpose(1, 2, 0)
                    .astype("uint8")
                )
                target = (
                    (target[0] * 255)
                    .detach()
                    .cpu()
                    .numpy()
                    .transpose(1, 2, 0)
                    .astype("uint8")
                )
                ssims.append(
                    calculate_ssim(target, output, data_range=255, multichannel=True,)
                )
        
        gt_median = np.median(target_ims)

        print("Test MAE loss: ", np.mean(MAEs / gt_median), " ", "\u00B1", " ", np.std(MAEs / gt_median))

        print(
            "Test SSIM: ", np.mean(ssims), " ", "\u00B1", " ", np.std(ssims),
        )

        return ssims, MAEs

    def infer_full_image(self, input, C_out, kernel_size=256, stride=128):
        self.model.eval()
        B, C, W, H = input.shape
        pad_W = kernel_size - W % kernel_size
        pad_H = kernel_size - H % kernel_size

        x, _, _ = compute_pyramid_patch_weight_loss(kernel_size, kernel_size)

        input = F.pad(input, (0, pad_H, 0, pad_W), mode="reflect").squeeze(0)
        _, W_pad, H_pad = input.shape
        patches = input.unfold(1, kernel_size, stride).unfold(2, kernel_size, stride)

        c, n_w, n_h, w, h = patches.shape
        patches = patches.contiguous().view(c, -1, kernel_size, kernel_size)

        dataset = torch.utils.data.TensorDataset(patches.permute(1, 0, 2, 3))
        batch_size = 8
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        op = []
        for batch_idx, sample1 in enumerate(dataloader):
            patch_op = self.model(sample1[0])
            op.append(patch_op)
        op = torch.cat(op).permute(1, 0, 2, 3)

        op = op.permute(0, 2, 3, 1).reshape(1, -1, n_w * n_h)
        weights_op = (
            torch.from_numpy(x)
            .unsqueeze(0)
            .unsqueeze(-1)
            .repeat(1, C_out, 1, n_w * n_h)
            .reshape(1, -1, n_w * n_h)
        ).cuda()
        op = torch.mul(weights_op, op)
        op = F.fold(
            op,
            output_size=(W_pad, H_pad),
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
        )
        weights_op = F.fold(
            weights_op,
            output_size=(W_pad, H_pad),
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
        )
        op = torch.div(op, weights_op)

        output = torch.clamp(op, 0.0, 1.0)
        output = output[:, :, :W, :H]
        return output

    def write_output_images(self, output, image_name, preprocess_step, preprocess_stats):
        if preprocess_step == "normalize":
            min = preprocess_stats[0].cuda()
            max = preprocess_stats[1].cuda()
            output = (
                ((max - min) * output + min)
                .cpu()
                .numpy()
                .transpose(1, 2, 0)
                .astype(np.uint16)
            )
        elif preprocess_step == "standardize":
            mean = preprocess_stats[0].cuda()
            std = preprocess_stats[1].cuda()
            output = (
                ((output * std) + mean).cpu().clamp(0, 65535).numpy().transpose(1, 2, 0).astype(np.uint16)
            )
        else:
            output = (output * 65535).cpu().numpy().transpose(1, 2, 0).astype(np.uint16)
        for i, filename in enumerate(image_name):
            cv2.imwrite(
                os.path.join(self.image_folder, f"{filename}"), output[:, :, i],
            )


def compute_pyramid_patch_weight_loss(width: int, height: int) -> np.ndarray:
    """Compute a weight matrix that assigns bigger weight on pixels in center and
    less weight to pixels on image boundary.
    This weight matrix then used for merging individual tile predictions and helps dealing
    with prediction artifacts on tile boundaries.
    :param width: Tile width
    :param height: Tile height
    :return: Since-channel image [Width x Height]
    """
    xc = width * 0.5
    yc = height * 0.5
    xl = 0
    xr = width
    yb = 0
    yt = height
    Dc = np.zeros((width, height))
    De = np.zeros((width, height))

    Dcx = np.square(np.arange(width) - xc + 0.5)
    Dcy = np.square(np.arange(height) - yc + 0.5)
    Dc = np.sqrt(Dcx[np.newaxis].transpose() + Dcy)

    De_l = np.square(np.arange(width) - xl + 0.5) + np.square(0.5)
    De_r = np.square(np.arange(width) - xr + 0.5) + np.square(0.5)
    De_b = np.square(0.5) + np.square(np.arange(height) - yb + 0.5)
    De_t = np.square(0.5) + np.square(np.arange(height) - yt + 0.5)

    De_x = np.sqrt(np.minimum(De_l, De_r))
    De_y = np.sqrt(np.minimum(De_b, De_t))
    De = np.minimum(De_x[np.newaxis].transpose(), De_y)

    alpha = (width * height) / np.sum(np.divide(De, np.add(Dc, De)))
    W = alpha * np.divide(De, np.add(Dc, De))
    return W, Dc, De
