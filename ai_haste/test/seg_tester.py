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


class PureSegTester:
    def __init__(self, config, save_folder, save_softmax=False):
        self.config = config
        self.save_folder = save_folder
        self.model_checkpoint = torch.load(
            os.path.join(save_folder, config["model_path"])
        )
        self.image_folder = os.path.join(self.save_folder, "test_images")
        if not os.path.exists(self.image_folder):
            os.makedirs(self.image_folder)
        self.save_softmax = save_softmax
        if self.save_softmax:
            self.softmax_save_folder = os.path.join(self.image_folder, "softmax")
            if not os.path.exists(self.softmax_save_folder):
                os.makedirs(self.softmax_save_folder)

    def test(self, dataloader, models):
        # self.models = []
        # for model in models:
        model = models[0]
        model = model.cuda()
        self.model = torch.nn.DataParallel(model)

        # self.model = self.models[0]
        # self.model.module.load_state_dict(self.model_checkpoint)
        self.model.load_state_dict(self.model_checkpoint["model_state_dict"])
        self.model.eval()

        best_train_epoch = self.model_checkpoint["epoch"]
        best_train_loss = self.model_checkpoint["epoch_loss"]
        print(f"Epoch:{best_train_epoch},Loss:{best_train_loss}")

        test_iou= self._test_epoch(dataloader)

        df = pd.DataFrame({"IOU": test_iou})
        df.to_csv(os.path.join(self.save_folder, "losses_test.csv"), index=False)

    def _test_epoch(self, dataloader):
        ious = []
        with torch.no_grad():
            for batch_idx, sample in enumerate(tqdm(dataloader)):
                image_name = sample[2]
                preprocess_step = sample[3]
                preprocess_stats = sample[4]
                magnification = sample[5]

                input = sample[0].cuda().to(non_blocking=True)
                mask = sample[1][:, 0].unsqueeze(1).cuda().to(non_blocking=True)
                mask_onehot = F.one_hot(mask.long()).squeeze(1).permute(0, 3, 1, 2)
                C_mask_out = mask_onehot.shape[1]
                target = sample[1][:, 1].unsqueeze(1).cuda().to(non_blocking=True)
                C_out = target.shape[1]

                output_mask, mask_op_softmax = self.infer_full_image(
                    input, C_mask_out, kernel_size=512, stride=256
                )

                if self.save_softmax:
                    np.save(
                        os.path.join(
                            self.softmax_save_folder, f"softmax_{image_name[0]}",
                        ),
                        mask_op_softmax.astype(np.float32),
                    )
                    np.save(
                        os.path.join(self.softmax_save_folder, f"mask_{image_name[0]}",),
                        mask.cpu()
                        .squeeze(0)
                        .numpy()
                        .transpose(1, 2, 0)
                        .astype(np.float32),
                    )

                intersection = torch.logical_and(mask, output_mask)
                union = torch.logical_or(mask, output_mask)
                iou = torch.sum(intersection) / torch.sum(union)
                ious.append(iou.item())


                self.write_output_images(
                    target[0],
                    output_mask[0],
                    mask[0],
                    image_name,
                    preprocess_step[0],
                    preprocess_stats,
                    magnification[0],
                )

        print(
            "Test IOUs: ", np.mean(ious), " ", "\u00B1", " ", np.std(ious),
        )

        return ious

    def infer_full_image(self, input, C_mask_out, kernel_size=256, stride=128):
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
        batch_size = 4
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        mask_op = []
        for batch_idx, sample1 in enumerate(dataloader):
            patch_mask_op = self.model(sample1[0])
            mask_op.append(patch_mask_op)
        mask_op = torch.cat(mask_op).permute(1, 0, 2, 3)

        mask_op = mask_op.permute(0, 2, 3, 1).reshape(1, -1, n_w * n_h)

        weights_mask_op = (
            torch.from_numpy(x)
            .unsqueeze(0)
            .unsqueeze(-1)
            .repeat(1, C_mask_out, 1, n_w * n_h)
            .reshape(1, -1, n_w * n_h)
        ).cuda()
        mask_op = torch.mul(weights_mask_op, mask_op)
        mask_op = F.fold(
            mask_op,
            output_size=(W_pad, H_pad),
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
        )
        weights_mask_op = F.fold(
            weights_mask_op,
            output_size=(W_pad, H_pad),
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
        )
        mask_op = torch.div(mask_op, weights_mask_op)

        mask_sftmax_op = mask_op[:, :, :W, :H]
        mask_op = mask_op.argmax(dim=1).unsqueeze(1)
        mask_output = mask_op[:, :, :W, :H]
        return mask_output, mask_sftmax_op


    def write_output_images(
        self,
        target,
        output_mask,
        target_mask,
        image_name,
        preprocess_step,
        preprocess_stats,
        magnification,
    ):
        image_save_folder = os.path.join(self.image_folder, f"{magnification}_images")
        if not os.path.exists(image_save_folder):
            os.makedirs(image_save_folder)
        mask_save_folder = os.path.join(self.image_folder, f"{magnification}_images")
        if not os.path.exists(mask_save_folder):
            os.makedirs(mask_save_folder)

        if preprocess_step == "normalize":
            min = preprocess_stats[0].cuda()
            max = preprocess_stats[1].cuda()
            target = (
                ((max - min) * target + min)
                .cpu()
                .numpy()
                .transpose(1, 2, 0)
                .astype(np.uint16)
            )
        elif preprocess_step == "standardize":
            mean = preprocess_stats[0].cuda()
            std = preprocess_stats[1].cuda()
            target = (
                ((target * std) + mean).cpu().numpy().transpose(1, 2, 0).astype(np.uint16)
            )
        else:
            target = (target * 65535).cpu().numpy().transpose(1, 2, 0).astype(np.uint16)
        for i, filename in enumerate(image_name):
            cv2.imwrite(
                os.path.join(image_save_folder, f"{filename}"), target[:, :, i],
            )
            cv2.imwrite(
                os.path.join(mask_save_folder, f"mask_{filename}"),
                (output_mask.cpu().numpy().transpose(1, 2, 0)[:, :, i] * 65535).astype(
                    np.uint16
                ),
            )
            cv2.imwrite(
                os.path.join(image_save_folder, f"target_mask_{filename}"),
                (target_mask.cpu().numpy().transpose(1, 2, 0)[:, :, i] * 65535).astype(
                    np.uint16
                ),
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
