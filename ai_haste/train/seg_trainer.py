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
import matplotlib.pyplot as plt

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

class SegTrainer:
    def __init__(self, config, save_folder):
        self.config = config
        self.epochs = config["epochs"]
        self.optimizer_config = config["optimizer"]
        self.lr_scheduler_config = config["lr_scheduler"]
        self.save_folder = save_folder
        # if config["save_name"] != "":
        #     self.save_folder = self.save_folder + "_" + config["save_name"]
        # if not os.path.exists(self.save_folder):
        #     os.makedirs(self.save_folder)
        self.save_interval = config["save_interval"]

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
        optimizer = getattr(optim, self.optimizer_config["type"])(**self.optimizer_config["args"])
        if self.lr_scheduler_config:
            self.lr_scheduler_config["args"]["optimizer"] = optimizer
            self.lr_scheduler = getattr(optim.lr_scheduler, self.lr_scheduler_config["type"])(
                **self.lr_scheduler_config["args"]
            )

        # criterion = criterions[0]
        all_train_losses_log = [[] for i in range(len(criterions))]
        all_valid_losses_log = [[] for i in range(len(criterions))]
        best_loss = np.inf

        for epoch in range(1, self.epochs):
            train_losses = self._train_epoch(train_dataloader, optimizer, criterions, epoch)
            for i in range(len(all_train_losses_log)):
                all_train_losses_log[i].extend(train_losses[i])

                df_train = pd.DataFrame(all_train_losses_log).transpose()
                df_train.columns = [x["loss"].__class__.__name__ for x in criterions]
                df_train.to_csv(os.path.join(self.save_folder, "losses_train.csv"), index=False)

            if epoch % 5 == 0:
                valid_losses, epoch_loss = self._valid_epoch(valid_dataloader, criterions, epoch)
                for i in range(len(all_valid_losses_log)):
                    all_valid_losses_log[i].extend(valid_losses[i])

                df_valid = pd.DataFrame(all_valid_losses_log).transpose()
                df_valid.columns = [x["loss"].__class__.__name__ for x in criterions]
                df_valid.to_csv(os.path.join(self.save_folder, "losses_valid.csv"), index=False)
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

            if self.lr_scheduler:
                self.lr_scheduler.step()

        return self.save_folder
        # return self.model

    def _train_epoch(self, dataloader, optimizer, criterions, epoch):
        outer = tqdm(total=len(dataloader), desc="Batches Processed:", position=0)
        running_loss_desc = tqdm(total=0, position=1, bar_format="{desc}")
        total_loss_desc = tqdm(total=0, position=2, bar_format="{desc}")
        total_loss = 0.0
        running_total_loss = 0.0
        running_losses = [0.0 for i in range(len(criterions))]
        total_losses = [0.0 for i in range(len(criterions))]
        log_interval = 2
        loss_log = [[] for i in range(len(criterions))]

        running_seg_loss = 0.0
        total_seg_loss = 0.0

        #criterion_seg = torch.nn.CrossEntropyLoss()

        criterion_seg = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(10.))

        self.model.train()

        for batch_idx, sample in enumerate(dataloader):
            

            # plot_multi(3,3,[sample[0][0][i] for i in range(9)])

            # plt.imshow(sample[1][:,0][0])
            # plt.show()
            input = sample[0].cuda().to(non_blocking=True)
            target = sample[1][:,0].unsqueeze(1).cuda().to(non_blocking=True)
            mask = sample[1][:,1]
            mask = to_one_hot(mask.long(), 2).permute(0,3,1,2).cuda().to(non_blocking=True)

            optimizer.zero_grad()

            output, output_seg = self.model(input)
            
            #output = torch.clamp(output, 0.0, 1.0)


            losses = []
            loss = 0
            for i, criterion in enumerate(criterions):
                loss_class = criterion["loss"](output, target)
                running_losses[i] += loss_class.item()
                total_losses[i] += loss_class.item()
                loss_log[i].append(loss_class.item())
                losses.append(loss_class.item())
                loss += criterion["weight"] * loss_class

            loss_seg = criterion_seg(output_seg, mask.float())

            running_seg_loss += loss_seg.item()
            total_seg_loss += loss_seg.item()
            
            loss = loss +  0.05 * loss_seg

            loss.backward()
            optimizer.step()

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
                loss_desc += (
                        f" Current {'Seg_loss'} {running_seg_loss/log_interval:.5f}"
                    )
                running_loss_desc.set_description_str(loss_desc)
                running_total_loss = 0.0
                running_seg_loss = 0.0
        if epoch % self.save_interval == 0:
            torch.save(
                self.model.module.state_dict(),
                os.path.join(self.save_folder, f"model_epoch_{epoch}.pth"),
            )


        loss_desc = f"Train Epoch {epoch} Total Loss:{total_loss/len(dataloader):.5f}"
        for i, criterion in enumerate(criterions):
            loss_name = criterion["loss"].__class__.__name__
            loss_desc += f" Total {loss_name} {total_losses[i]/len(dataloader):.5f}"
        total_loss_desc.set_description_str(loss_desc)

        return loss_log


    def _valid_epoch(self, dataloader, criterions, epoch):
        self.model.eval()
        outer = tqdm(total=len(dataloader), desc="Batches Processed:", position=0)
        total_loss_desc = tqdm(total=0, position=2, bar_format="{desc}")
        total_loss = 0.0
        log_interval = 1
        loss_log = [[] for i in range(len(criterions))]
        with torch.no_grad():
            for batch_idx, sample in enumerate(dataloader):
                image_name = sample[2]
                preprocess_step = sample[3]
                preprocess_stats = sample[4]

                input = sample[0].cuda().to(non_blocking=True)
                target = sample[1][:,0].unsqueeze(1).cuda().to(non_blocking=True)
                mask = sample[1][:,1].cuda().to(non_blocking=True)

                C_out = target.shape[1]

                output, output_seg = self.infer_full_image(input, C_out, mask)

                losses = []
                loss = 0
                for i, criterion in enumerate(criterions):
                    loss_class = criterion["loss"](output, target)
                    loss_log[i].append(loss_class.item())
                    losses.append(loss_class.item())
                    loss += criterion["weight"] * loss_class

                total_loss += loss.item()

                if epoch % 10 == 0 and batch_idx == 0:
                    self.write_output_images(
                        output[0],
                        output_seg[0][0],
                        mask[0],
                        target[0],
                        image_name,
                        preprocess_step[0],
                        preprocess_stats,
                        epoch,
                    )

                outer.update(1)
        
        print('Percentage of non zero el.: ', torch.count_nonzero(output.detach().cpu())/output.detach().cpu().numel(), ' %')

        total_loss_desc.set_description_str(
            f"Valid Epoch {epoch} Total Loss:{total_loss/len(dataloader):.5f}"
        )
        return loss_log, total_loss / len(dataloader)

    def infer_full_image(self, input, C_out, mask, kernel_size=256, stride=256):
        B, C, W, H = input.shape
        pad_W = kernel_size - W % kernel_size
        pad_H = kernel_size - H % kernel_size

        input = F.pad(input, (0, pad_H, 0, pad_W), mode="reflect").squeeze(0)
        patches = input.unfold(1, kernel_size, stride).unfold(2, kernel_size, stride)

        c, n_w, n_h, w, h = patches.shape
        patches = patches.contiguous().view(c, -1, kernel_size, kernel_size)

        dataset = torch.utils.data.TensorDataset(patches.permute(1, 0, 2, 3))
        batch_size = 8
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        op = []
        seg = []


        for batch_idx, sample1 in enumerate(dataloader):
            op_out, seg_out = self.model(sample1[0])
            op.append(op_out)
            seg.append(seg_out)
        op = torch.cat(op).permute(1, 0, 2, 3)
        seg = torch.cat(seg).permute(1, 0, 2, 3)

        op = op.view(C_out, n_w, n_h, w, h)
        output_h = n_w * w
        output_w = n_h * h
        op = op.permute(0, 1, 3, 2, 4).contiguous()
        op = op.view(C_out, output_h, output_w)

        seg = torch.argmax(torch.nn.Sigmoid()(seg), dim=0).unsqueeze(0)
        seg = seg.view(C_out, n_w, n_h, w, h)
        seg = seg.permute(0, 1, 3, 2, 4).contiguous()
        seg = seg.view(C_out, output_h, output_w)

        #output = torch.clamp(op, 0.0, 1.0)
        output = op[:, :W, :H].unsqueeze(0)
        #output_seg = torch.argmax(torch.nn.Sigmoid()(seg), dim=1)
        output_seg = seg[:, :W, :H].unsqueeze(0)
        return output, output_seg

    def write_output_images(
        self, output, output_seg, mask, target, image_name, preprocess_step, preprocess_stats, epoch
    ):

        output = output / 255
        target = target / 255

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
                os.path.join(self.image_folder, f"epoch_{epoch}_output_{filename}"),
                output[:, :, i],
            )
            cv2.imwrite(
                os.path.join(self.image_folder, f"epoch_{epoch}_target_{filename}"),
                target[:, :, i],
            )

            cv2.imwrite(
                os.path.join(self.image_folder, f"epoch_{epoch}_target_mask_{filename}.tif"),
                        (mask.detach().cpu().numpy()*255).astype(np.uint8))

            cv2.imwrite(
                os.path.join(self.image_folder, f"epoch_{epoch}_output_seg_{filename}.tif"),
                        (output_seg.detach().cpu().numpy()*255).astype(np.uint8))