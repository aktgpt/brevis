import os
import random

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm import tqdm

sns.set_theme(style="darkgrid")
mpl.rcParams["agg.path.chunksize"] = 10000

def plot_multi(cols, rows, images, **kwargs):
    fig, axs = plt.subplots(rows, cols)
    axs = axs.ravel()

    for ax, im in zip(axs, images):
        ax.imshow(im, **kwargs)
 
    plt.show(block=True)

def to_one_hot(y, num_classes):
    scatter_dim = len(y.size())
    y_tensor = y.view(*y.size(), -1)
    zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)
        
    return zeros.scatter(scatter_dim, y_tensor, 1)

class PureSegTrainer:
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
        self.resume_training = config["resume_training"]
        if self.resume_training:
            self.model_checkpoint = torch.load(config["model_path"])

    def train(self, train_dataloader, valid_dataloader, models, criterions):
        self.models = []
        for model in models:
            model = model.cuda()
            model.summary()
            self.models.append(torch.nn.DataParallel(model))

        self.model = self.models[0]
        if self.resume_training:
            self.model.load_state_dict(self.model_checkpoint["model_state_dict"])

        self.optimizer_config["args"]["params"] = self.model.parameters()
        self.optimizer = getattr(optim, self.optimizer_config["type"])(
            **self.optimizer_config["args"]
        )
        if self.lr_scheduler_config:
            self.lr_scheduler_config["args"]["optimizer"] = self.optimizer
            self.lr_scheduler = getattr(
                optim.lr_scheduler, self.lr_scheduler_config["type"]
            )(**self.lr_scheduler_config["args"])
        else:
            self.lr_scheduler = False

        all_train_losses_log = [[] for i in range(len(criterions))]
        all_valid_losses_log = [[] for i in range(len(criterions)+1)]
        best_loss = np.inf

        for epoch in range(1, self.epochs + 1):
            train_losses, epoch_loss = self._train_epoch(
                train_dataloader, criterions, epoch
            )
            for i in range(len(all_train_losses_log)):
                all_train_losses_log[i].extend(train_losses[i])
            self.plot_train_losses(all_train_losses_log, criterions)

            if epoch % 5 == 0:
                valid_losses, epoch_loss = self._valid_epoch(valid_dataloader, criterions, epoch)
                for i in range(len(all_valid_losses_log)):
                    all_valid_losses_log[i].extend(valid_losses[i])

                self.plot_valid_losses(all_valid_losses_log, criterions)
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
                print(f"Saving best loss model with loss:{epoch_loss}")
            # if self.lr_scheduler:
            #     self.lr_scheduler.step()
        return self.save_folder

    def plot_valid_losses(self, all_valid_losses_log, criterions):
        df_valid = pd.DataFrame(all_valid_losses_log).transpose()
        df_cols = ["IOU"]
        df_cols.extend([x["loss"].__class__.__name__ for x in criterions])
        df_valid.columns = df_cols
        df_valid.to_csv(os.path.join(self.save_folder, "losses_valid.csv"), index=False)
        idx = np.arange(0, len(df_valid), 1)
        for i in list(df_valid):
            sns.lineplot(x=idx, y=df_valid[i])
            # plt.ylim((0, 2))
            plt.savefig(os.path.join(self.save_folder, f"valid_{i}.png"), dpi=1000)
            plt.close()

    def plot_train_losses(self, all_train_losses_log, criterions):
        df_train = pd.DataFrame(all_train_losses_log).transpose()
        df_train.columns = [x["loss"].__class__.__name__ for x in criterions]
        df_train.to_csv(os.path.join(self.save_folder, "losses_train.csv"), index=False)
        idx = np.arange(0, len(df_train), 1)
        for i in list(df_train):
            sns.lineplot(x=idx, y=df_train[i])
            # plt.ylim((0, 2))
            plt.savefig(os.path.join(self.save_folder, f"train_{i}.png"), dpi=1000)
            plt.close()

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

        log_interval = 20
        loss_log = [[] for i in range(len(criterions))]

        iters = len(dataloader)
        for batch_idx, sample in enumerate(dataloader):
            input = sample[0].cuda().to(non_blocking=True)
            mask = sample[1][:, 0].cuda().to(non_blocking=True)
            mask_onehot = (
                F.one_hot(mask.unsqueeze(1).long()).squeeze(1).permute(0, 3, 1, 2)
            )

            self.optimizer.zero_grad()

            mask_op = self.model(input)
            loss = 0

            loss_segmentation = criterions[0]["loss"](mask_op, mask_onehot.float())
            running_losses[0] += loss_segmentation.item()
            total_losses[0] += loss_segmentation.item()
            loss += criterions[0]["weight"] * loss_segmentation


            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
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
                    loss_log[i].append(running_losses[i] / log_interval)
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
                {
                    "model_state_dict": self.model.state_dict(),
                    "epoch": epoch,
                    "epoch_loss": total_loss / iters,
                },
                os.path.join(self.save_folder, f"model_epoch_{epoch}.pth"),
            )

        return loss_log, total_loss / iters


    def _valid_epoch(self, dataloader, criterions, epoch):
        outer = tqdm(total=len(dataloader), desc="Batches Processed:", position=0)
        total_loss_desc = tqdm(total=0, position=2, bar_format="{desc}")
        total_loss = 0.0
        total_bce_loss = 0.0
        loss_log = [[] for i in range(len(criterions) + 1)]

        iters = len(dataloader)
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

                output_mask, output_mask_softmax = self.infer_full_image(
                    input,C_mask_out, kernel_size=512, stride=256
                )


                intersection = torch.logical_and(mask, output_mask)
                union = torch.logical_or(mask, output_mask)
                loss_segmentation = torch.sum(intersection) / torch.sum(union)

                loss_log[0].append(loss_segmentation.item())

                
                loss_bce = criterions[0]["loss"](output_mask_softmax, mask_onehot.float())
                total_bce_loss += loss_bce.item()
                loss_log[1].append(loss_bce.item())

                total_loss += loss_bce.item()

                if batch_idx == 0:
                    self.write_output_images(
                        output_mask[0],
                        mask[0],
                        image_name,
                        magnification[0],
                        epoch,
                    )

                outer.update(1)

        total_loss_desc.set_description_str(
            f"Valid Epoch {epoch} Total Loss:{total_loss/iters:.5f}"
        )
        return loss_log, total_loss / iters
    
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
        output_mask,
        target_mask,
        image_name,
        magnification,
        epoch,
    ):
        image_save_folder = os.path.join(self.image_folder, f"{magnification}_images")
        if not os.path.exists(image_save_folder):
            os.makedirs(image_save_folder)

        for i, filename in enumerate(image_name):
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
