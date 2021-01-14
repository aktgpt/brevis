import os
import random

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm import tqdm


class LUPISegmentationTrainer:
    def __init__(self, config, save_folder):
        self.config = config
        self.epochs = config["epochs"]
        self.optimizer_config = config["optimizer"]
        self.lr_scheduler_config = config["lr_scheduler"]
        self.save_folder = save_folder
        self.save_interval = config["save_interval"]
        self.gradient_accumulation = config["gradient_accumulation"]
        self.image_folder = os.path.join(self.save_folder, "images")
        if not os.path.exists(self.image_folder):
            os.makedirs(self.image_folder)

    def train(self, train_dataloader, valid_dataloader, models, criterions):
        self.models = []
        for model in models:
            model = model.cuda()
            self.models.append(torch.nn.DataParallel(model))

        self.model = self.models[0]
        self.optimizer_config["args"]["params"] = self.model.parameters()
        self.optimizer = getattr(optim, self.optimizer_config["type"])(
            **self.optimizer_config["args"]
        )
        if self.lr_scheduler_config:
            self.lr_scheduler_config["args"]["optimizer"] = self.optimizer
            self.lr_scheduler = getattr(
                optim.lr_scheduler, self.lr_scheduler_config["type"]
            )(**self.lr_scheduler_config["args"])

        all_train_losses_log = [[] for i in range(len(criterions))]
        all_valid_losses_log = [[] for i in range(len(criterions))]
        best_loss = np.inf

        for epoch in range(1, self.epochs + 1):
            train_losses = self._train_epoch(train_dataloader, criterions, epoch)
            for i in range(len(all_train_losses_log)):
                all_train_losses_log[i].extend(train_losses[i])

                df_train = pd.DataFrame(all_train_losses_log).transpose()
                df_train.columns = [x["loss"].__class__.__name__ for x in criterions]
                df_train.to_csv(
                    os.path.join(self.save_folder, "losses_train.csv"), index=False
                )

            if epoch % 1 == 0:
                valid_losses, epoch_loss = self._valid_epoch(
                    valid_dataloader, criterions, epoch
                )
                for i in range(len(all_valid_losses_log)):
                    all_valid_losses_log[i].extend(valid_losses[i])

                df_valid = pd.DataFrame(all_valid_losses_log).transpose()
                df_cols = ["IOU"]
                df_cols.extend([x["loss"].__class__.__name__ for x in criterions[1:]])
                df_valid.columns = df_cols
                df_valid.to_csv(
                    os.path.join(self.save_folder, "losses_valid.csv"), index=False
                )
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    torch.save(
                        {
                            "model_state_dict": self.model.state_dict(),
                            "epoch": epoch,
                            "epoch_loss": epoch_loss,
                        },
                        os.path.join(self.save_folder, f"model_best_loss.pth"),
                    )

        return self.save_folder

    def _train_epoch(self, dataloader, criterions, epoch):
        self.model.train()
        outer = tqdm(total=len(dataloader), desc="Batches Processed:", position=0)
        running_loss_desc = tqdm(total=0, position=1, bar_format="{desc}")
        total_loss_desc = tqdm(total=0, position=2, bar_format="{desc}")
        learning_rate_desc = tqdm(total=0, position=2, bar_format="{desc}")
        total_loss = 0.0
        total_losses = [0.0 for i in range(len(criterions))]
        running_total_loss = 0.0
        running_losses = [0.0 for i in range(len(criterions))]

        log_interval = 10
        loss_log = [[] for i in range(len(criterions))]

        iters = len(dataloader)
        for batch_idx, sample in enumerate(dataloader):
            input = sample[0].cuda().to(non_blocking=True)
            mask = sample[1][:, 0].cuda().to(non_blocking=True)
            mask_onehot = (
                F.one_hot(mask.unsqueeze(1).long()).squeeze(1).permute(0, 3, 1, 2)
            )
            target = sample[1][:, 1].unsqueeze(1).cuda().to(non_blocking=True)

            self.optimizer.zero_grad()

            mask_op, output = self.model(input)
            output = torch.clamp(output, 0.0, 1.0)
            losses = []
            loss = 0

            loss_segmentation = criterions[0]["loss"](mask_op, mask_onehot.float())
            running_losses[0] += loss_segmentation.item()
            total_losses[0] += loss_segmentation.item()
            loss_log[0].append(loss_segmentation.item())
            losses.append(loss_segmentation.item())
            loss += criterions[0]["weight"] * loss_segmentation

            for i, criterion in enumerate(criterions[1:]):
                loss_class = criterion["loss"](output, target)
                running_losses[i + 1] += loss_class.item()
                total_losses[i + 1] += loss_class.item()
                loss_log[i + 1].append(loss_class.item())
                losses.append(loss_class.item())
                loss += criterion["weight"] * loss_class

            loss.backward()
            self.optimizer.step()
            if self.lr_scheduler:
                self.lr_scheduler.step(epoch - 1 + batch_idx / iters)
                learning_rate_desc.set_description_str(
                    "Epoch-{0} lr: {1}".format(
                        epoch, self.optimizer.param_groups[0]["lr"]
                    )
                )

            total_loss += loss.item()
            running_total_loss += loss.item()

            outer.update(1)
            if (batch_idx + 1) % log_interval == 0:
                loss_desc = f"Train Epoch {epoch} Current Total Loss:{running_total_loss/log_interval:.5f}"
                for i, criterion in enumerate(criterions):
                    loss_name = criterion["loss"].__class__.__name__
                    loss_desc += (
                        f" Current {loss_name} {running_losses[i]/log_interval:.5f}"
                    )
                    running_losses[i] = 0.0
                running_loss_desc.set_description_str(loss_desc)
                running_total_loss = 0.0

        loss_desc = f"Train Epoch {epoch} Total Loss:{total_loss/iters:.5f}"
        for i, criterion in enumerate(criterions):
            loss_name = criterion["loss"].__class__.__name__
            loss_desc += f" Total {loss_name} {total_losses[i]/iters:.5f}"
        total_loss_desc.set_description_str(loss_desc)

        if epoch % self.save_interval == 0:
            torch.save(
                self.model.module.state_dict(),
                os.path.join(self.save_folder, f"model_epoch_{epoch}.pth"),
            )

        return loss_log

    def _valid_epoch(self, dataloader, criterions, epoch):
        outer = tqdm(total=len(dataloader), desc="Batches Processed:", position=0)
        total_loss_desc = tqdm(total=0, position=2, bar_format="{desc}")
        total_loss = 0.0
        loss_log = [[] for i in range(len(criterions))]
        with torch.no_grad():
            for batch_idx, sample in enumerate(dataloader):
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

                output_mask, output = self.infer_full_image(
                    input, C_out, C_mask_out, kernel_size=256, stride=128
                )

                losses = []
                loss = 0

                intersection = torch.logical_and(mask, output_mask)
                union = torch.logical_or(mask, output_mask)
                loss_segmentation = torch.sum(intersection) / torch.sum(union)

                loss_log[0].append(loss_segmentation.item())
                losses.append(loss_segmentation.item())
                loss += loss_segmentation

                for i, criterion in enumerate(criterions[1:]):
                    loss_class = criterion["loss"](output, target)
                    loss_log[i + 1].append(loss_class.item())
                    losses.append(loss_class.item())
                    loss += criterion["weight"] * loss_class

                total_loss += loss.item()

                if epoch % 1 == 0 and batch_idx % 5 == 0:
                    self.write_output_images(
                        output[0],
                        target[0],
                        output_mask[0],
                        mask[0],
                        image_name,
                        preprocess_step[0],
                        preprocess_stats,
                        magnification[0],
                        epoch,
                    )

                outer.update(1)

        total_loss_desc.set_description_str(
            f"Valid Epoch {epoch} Total Loss:{total_loss/len(dataloader):.5f}"
        )
        return loss_log, total_loss / len(dataloader)

    def infer_full_image(self, input, C_out, C_mask_out, kernel_size=256, stride=256):
        with torch.no_grad():
            self.model.eval()
            B, C, W, H = input.shape
            pad_W = kernel_size - W % kernel_size
            pad_H = kernel_size - H % kernel_size

            input = F.pad(input, (0, pad_H, 0, pad_W), mode="reflect").squeeze(0)
            _, W_pad, H_pad = input.shape
            patches = input.unfold(1, kernel_size, stride).unfold(2, kernel_size, stride)

            c, n_w, n_h, w, h = patches.shape
            patches = patches.contiguous().view(c, -1, kernel_size, kernel_size)

            dataset = torch.utils.data.TensorDataset(patches.permute(1, 0, 2, 3))
            batch_size = 8
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
            weights = torch.ones_like(op)
            op = F.fold(
                op,
                output_size=(W_pad, H_pad),
                kernel_size=(kernel_size, kernel_size),
                stride=(stride, stride),
            )
            mask_op = F.fold(
                mask_op,
                output_size=(W_pad, H_pad),
                kernel_size=(kernel_size, kernel_size),
                stride=(stride, stride),
            )
            weights = F.fold(
                weights,
                output_size=(W_pad, H_pad),
                kernel_size=(kernel_size, kernel_size),
                stride=(stride, stride),
            )
            op = torch.div(op, weights)
            mask_op = torch.div(mask_op, weights)

            output = torch.clamp(op, 0.0, 1.0)
            mask_op = mask_op.argmax(dim=1).unsqueeze(1)
            output = output[:, :, :W, :H]
            mask_output = mask_op[:, :, :W, :H]
        return mask_output, output

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
        epoch,
    ):
        image_save_folder = os.path.join(self.image_folder, f"{magnification}_images")
        if not os.path.exists(image_save_folder):
            os.makedirs(image_save_folder)

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
                ((output * std) + mean).cpu().numpy().transpose(1, 2, 0).astype(np.uint16)
            )
            target = (
                ((target * std) + mean).cpu().numpy().transpose(1, 2, 0).astype(np.uint16)
            )
        else:
            output = (output * 65535).cpu().numpy().transpose(1, 2, 0).astype(np.uint16)
            target = (target * 65535).cpu().numpy().transpose(1, 2, 0).astype(np.uint16)
        for i, filename in enumerate(image_name):
            cv2.imwrite(
                os.path.join(image_save_folder, f"epoch_{epoch}_output_{filename}"),
                output[:, :, i],
            )
            cv2.imwrite(
                os.path.join(image_save_folder, f"epoch_{epoch}_target_{filename}"),
                target[:, :, i],
            )
            cv2.imwrite(
                os.path.join(image_save_folder, f"epoch_{epoch}_output_mask_{filename}"),
                (output_mask.cpu().numpy().transpose(1, 2, 0)[:, :, i] * 65535).astype(
                    np.uint16
                ),
            )
            cv2.imwrite(
                os.path.join(image_save_folder, f"epoch_{epoch}_target_mask_{filename}"),
                (target_mask.cpu().numpy().transpose(1, 2, 0)[:, :, i] * 65535).astype(
                    np.uint16
                ),
            )

