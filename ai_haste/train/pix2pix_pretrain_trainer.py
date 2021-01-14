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
from torch import nn

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)

class Pix2PixPreTrainer:
    def __init__(self, config, save_folder):
        self.config = config
        self.epochs = config["epochs"]
        self.gen_optimizer_config = config["gen_optimizer"]
        self.gen_lr_scheduler_config = config["gen_lr_scheduler"]
        self.save_folder = save_folder
        self.save_interval = config["save_interval"]
        self.image_folder = os.path.join(self.save_folder, "images")
        
        self.n_losses = 1 # reconstruction loss
        self.loss_names = ["gen_loss"]
        
        if not os.path.exists(self.image_folder):
            os.makedirs(self.image_folder)

    def train(self, train_dataloader, valid_dataloader, models, criterions):
        self.models = []
        for model in models:
            model = model.cuda()
            self.models.append(torch.nn.DataParallel(model))

        self.gen = self.models[0]
        
        self.gen = self.gen.apply(weights_init)
        
        self.gen_optimizer_config["args"]["params"] = self.gen.parameters()
        self.gen_optimizer = getattr(optim, self.gen_optimizer_config["type"])(
            **self.gen_optimizer_config["args"]
        )
        if self.gen_lr_scheduler_config:
            self.gen_lr_scheduler_config["args"]["optimizer"] = self.gen_optimizer
            self.gen_lr_scheduler = getattr(
                optim.lr_scheduler, self.gen_lr_scheduler_config["type"]
            )(**self.gen_lr_scheduler_config["args"])
        all_train_losses_log = [[] for i in range(self.n_losses)]
        all_valid_losses_log = [[] for i in range(self.n_losses + 1)]
        best_loss = np.inf

        for epoch in range(1, self.epochs + 1):
            train_losses = self._train_epoch(
                train_dataloader, criterions, epoch
            )
            for i in range(len(all_train_losses_log)):
                all_train_losses_log[i].extend(train_losses[i])

                df_train = pd.DataFrame(all_train_losses_log).transpose()
                df_train.columns = self.loss_names
                df_train.to_csv(
                    os.path.join(self.save_folder, "losses_train.csv"), index=False
                )

            if epoch % 5 == 0:
                valid_losses, epoch_loss = self._valid_epoch(
                    valid_dataloader, criterions, epoch
                )
                for i in range(len(all_valid_losses_log)):
                    all_valid_losses_log[i].extend(valid_losses[i])

                df_valid = pd.DataFrame(all_valid_losses_log).transpose()
                df_valid.columns = [x["loss"].__class__.__name__ for x in criterions] # L1 and SSIM
                df_valid.to_csv(
                    os.path.join(self.save_folder, "losses_valid.csv"), index=False
                )
                
            if self.gen_lr_scheduler:
                self.gen_lr_scheduler.step()

        return self.save_folder

    def _train_epoch(self, dataloader, criterions, epoch):
        recon_criterion = criterions[0]["loss"]
        
        self.gen.train()
        
        outer = tqdm(total=len(dataloader), desc="Batches Processed:", position=0)
        running_loss_desc = tqdm(total=0, position=1, bar_format="{desc}")
        total_loss_desc = tqdm(total=0, position=2, bar_format="{desc}")
        
        total_losses = [0.0 for i in range(self.n_losses)]
        running_losses = [0.0 for i in range(self.n_losses)]
        
        log_interval = 2
        loss_log = [[] for i in range(self.n_losses)]

        for batch_idx, sample in enumerate(dataloader):
            input = sample[0].cuda().to(non_blocking=True)
            target = sample[1].cuda().to(non_blocking=True)
            
            self.gen_optimizer.zero_grad()
            fake_images = self.gen(input)
            gen_loss = recon_criterion(fake_images, target)
            
            running_losses[0] += gen_loss.item()
            total_losses[0] += gen_loss.item()
            loss_log[0].append(gen_loss.item())
            
            gen_loss.backward()
            self.gen_optimizer.step()
            
            outer.update(1)
            if (batch_idx + 1) % log_interval == 0:
                
                loss_name = "gen_loss"
                loss_desc = f" Current {loss_name}: {running_losses[0]/log_interval:.5f}"
                running_losses[0] = 0.0
                running_loss_desc.set_description_str(loss_desc)
        
        if epoch % self.save_interval == 0:
            torch.save(
                self.gen.module.state_dict(),
                os.path.join(self.save_folder, f"model_gen_epoch_{epoch}.pth"), # need different names for gen and disc
            )
            torch.save(
                self.disc.module.state_dict(),
                os.path.join(self.save_folder, f"model_disc_epoch_{epoch}.pth"),
            )

        return loss_log

    def _valid_epoch(self, dataloader, criterions, epoch):
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
                target = sample[1].cuda().to(non_blocking=True)
                C_out = target.shape[1]

                output = self.infer_full_image(input, C_out)

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
                        target[0],
                        image_name,
                        preprocess_step[0],
                        preprocess_stats,
                        epoch,
                    )

                outer.update(1)

        total_loss_desc.set_description_str(
            f"Valid Epoch {epoch} Total Loss:{total_loss/len(dataloader):.5f}"
        )
        return loss_log, total_loss / len(dataloader)

    def infer_full_image(self, input, C_out, kernel_size=256, stride=256):
        self.gen.eval()
        B, C, W, H = input.shape
        pad_W = kernel_size - W % kernel_size
        pad_H = kernel_size - H % kernel_size

        input = F.pad(input, (0, pad_H, 0, pad_W), mode="reflect").squeeze(0)
        patches = input.unfold(1, kernel_size, stride).unfold(2, kernel_size, stride)

        c, n_w, n_h, w, h = patches.shape
        patches = patches.contiguous().view(c, -1, kernel_size, kernel_size)

        dataset = torch.utils.data.TensorDataset(patches.permute(1, 0, 2, 3))
        batch_size = 2
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        op = []
        for batch_idx, sample1 in enumerate(dataloader):
            op.append(self.gen(sample1[0]))
        op = torch.cat(op).permute(1, 0, 2, 3)

        op = op.view(C_out, n_w, n_h, w, h)
        output_h = n_w * w
        output_w = n_h * h
        op = op.permute(0, 1, 3, 2, 4).contiguous()
        op = op.view(C_out, output_h, output_w)

        output = torch.clamp(op, 0.0, 1.0)
        output = output[:, :W, :H].unsqueeze(0)
        return output

    def write_output_images(
        self, output, target, image_name, preprocess_step, preprocess_stats, epoch
    ):
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
