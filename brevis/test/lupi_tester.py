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


class LUPITester:
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

        test_iou, test_ssim, test_loss = self._test_epoch(dataloader)

        df = pd.DataFrame({"IOU": test_iou, "SSIM": test_ssim, "MAE": test_loss})
        df.to_csv(os.path.join(self.save_folder, "losses_test.csv"), index=False)

    def _test_epoch(self, dataloader):
        MAE_criterion = torch.nn.L1Loss()

        ssims = []
        MAEs = []
        ious = []
        target_ims = []
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

                output_mask, output, mask_op_softmax = self.infer_full_image(
                    input, C_out, C_mask_out, kernel_size=512, stride=256
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
                iou = torch.true_divide(torch.sum(intersection), torch.sum(union))
                ious.append(iou.item())

                output_8bit = (
                    (output[0] * 255)
                    .detach()
                    .cpu()
                    .numpy()
                    .transpose(1, 2, 0)
                    .astype("uint8")
                )
                target_8bit = (
                    (target[0] * 255)
                    .detach()
                    .cpu()
                    .numpy()
                    .transpose(1, 2, 0)
                    .astype("uint8")
                )
                ssims.append(
                    calculate_ssim(
                        target_8bit, output_8bit, data_range=255, multichannel=True,
                    )
                )

                output, target = self.write_output_images(
                    output[0],
                    target[0],
                    output_mask[0],
                    mask[0],
                    image_name,
                    preprocess_step[0],
                    preprocess_stats,
                    magnification[0],
                )

                MAEs.append(
                    mean_absolute_error(
                            output[:, :, 0].astype("float32"),
                            target[:, :, 0].astype("float32"),
                        ).item()
                    )
                
                target_ims.append(target[:, :, 0].astype('float32'))

        gt_median = np.median(target_ims)
        print("Test MAE loss: ", np.mean(MAEs)/ gt_median, " ", "\u00B1", " ", np.std(MAEs)/ gt_median)

        print(
            "Test SSIMs: ", np.mean(ssims), " ", "\u00B1", " ", np.std(ssims),
        )
        print(
            "Test IOUs: ", np.mean(ious), " ", "\u00B1", " ", np.std(ious),
        )

        return ious, ssims, MAEs

    def infer_full_image(self, input, C_out, C_mask_out, kernel_size=256, stride=128):
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
        batch_size = 16
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        op = []
        mask_op = []
        for batch_idx, sample1 in enumerate(dataloader):
            patch_mask_op, patch_op = self.model(sample1[0])
            op.append(patch_op)
            mask_op.append(patch_mask_op)
        op = torch.cat(op).permute(1, 0, 2, 3)
        mask_op = torch.cat(mask_op).permute(1, 0, 2, 3)

        op = op.permute(0, 2, 3, 1).reshape(1, -1, n_w * n_h)
        mask_op = mask_op.permute(0, 2, 3, 1).reshape(1, -1, n_w * n_h)
        # weights = torch.ones_like(op)
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
        # op = op.view(C_out, n_w, n_h, w, h)
        # mask_op = mask_op.view(C_mask_out, n_w, n_h, w, h)

        # output_h = n_w * w
        # output_w = n_h * h
        # op = op.permute(0, 1, 3, 2, 4).contiguous()
        # mask_op = mask_op.permute(0, 1, 3, 2, 4).contiguous()

        # op = op.view(C_out, output_h, output_w)
        # mask_op = mask_op.view(C_mask_out, output_h, output_w)

        output = torch.clamp(op, 0.0, 1.0)
        mask_op_softmax = (
            mask_op[:, :, :W, :H].squeeze(0).cpu().numpy().transpose(1, 2, 0)
        )
        mask_op = mask_op.argmax(dim=1).unsqueeze(1)
        output = output[:, :, :W, :H]
        mask_output = mask_op[:, :, :W, :H]
        return mask_output, output, mask_op_softmax

    def write_output_images(
        self,
        output,
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
            output = (
                ((max - min) * output + min)
                .cpu()
                .numpy()
                .transpose(1, 2, 0)
                .astype(np.uint16)
            )
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
            output = (
                ((output * std) + mean).cpu().clamp(0, 65535).numpy().transpose(1, 2, 0).astype(np.uint16)
            )
            target = (
                ((target * std) + mean).cpu().clamp(0, 65535).numpy().transpose(1, 2, 0).astype(np.uint16)
            )
        else:
            output = (output * 65535).clamp(0, 65535).cpu().numpy().transpose(1, 2, 0).astype(np.uint16)
            target = (target * 65535).clamp(0, 65535).cpu().numpy().transpose(1, 2, 0).astype(np.uint16)
        for i, filename in enumerate(image_name):
            cv2.imwrite(
                os.path.join(image_save_folder, f"{filename}"), output[:, :, i],
            )
            # cv2.imwrite(
            #     os.path.join(mask_save_folder, f"mask_{filename}"),
            #     (output_mask.cpu().numpy().transpose(1, 2, 0)[:, :, i] * 65535).astype(
            #         np.uint16
            #     ),
            # )
        return output, target


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
