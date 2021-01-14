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

def disc_weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)

class Pix2PixTrainer:
    def __init__(self, config, save_folder):
        self.config = config
        self.epochs = config["epochs"]
        self.pretrained_model_path = config["pretrained_model_path"]
        self.gen_optimizer_config = config["gen_optimizer"]
        self.disc_optimizer_config = config["disc_optimizer"]
        self.gen_lr_scheduler_config = config["gen_lr_scheduler"]
        self.disc_lr_scheduler_config = config["disc_lr_scheduler"]
        self.save_folder = save_folder
        self.save_interval = config["save_interval"]
        self.image_folder = os.path.join(self.save_folder, "images")
        
        self.lambda_recon = config["lambda_recon"]
        self.n_losses = 3 # generator, discriminator and reconstruction losses
        self.loss_names = ["gen_loss", "disc_loss", "recon_loss"]
        
        if not os.path.exists(self.image_folder):
            os.makedirs(self.image_folder)

    def train(self, train_dataloader, valid_dataloader, models, criterions):
        self.models = []
        for model in models:
            model = model.cuda()
            self.models.append(torch.nn.DataParallel(model))

        self.gen = self.models[0]
        self.disc = self.models[1]
        pretrained_model = torch.load(self.pretrained_model_path)
        self.gen.load_state_dict(pretrained_model['model_state_dict'])
        self.disc = self.disc.apply(disc_weights_init)
        
        self.gen_optimizer_config["args"]["params"] = self.gen.parameters()
        self.gen_optimizer = getattr(optim, self.gen_optimizer_config["type"])(
            **self.gen_optimizer_config["args"]
        )
        self.disc_optimizer_config["args"]["params"] = self.disc.parameters()
        self.disc_optimizer = getattr(optim, self.disc_optimizer_config["type"])(
            **self.disc_optimizer_config["args"]
        )
        if self.gen_lr_scheduler_config:
            self.gen_lr_scheduler_config["args"]["optimizer"] = self.gen_optimizer
            self.gen_lr_scheduler = getattr(
                optim.lr_scheduler, self.gen_lr_scheduler_config["type"]
            )(**self.gen_lr_scheduler_config["args"])
        if self.disc_lr_scheduler_config:
            self.disc_lr_scheduler_config["args"]["optimizer"] = self.disc_optimizer
            self.disc_lr_scheduler = getattr(
                optim.lr_scheduler, self.disc_lr_scheduler_config["type"]
            )(**self.disc_lr_scheduler_config["args"])
        all_train_losses_log = [[] for i in range(self.n_losses)]
        all_valid_losses_log = [[] for i in range(len(criterions) - 1)]
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
                df_valid.columns = [x["loss"].__class__.__name__ for x in criterions[1:]]
                df_valid.to_csv(
                    os.path.join(self.save_folder, "losses_valid.csv"), index=False
                )
                
            if self.gen_lr_scheduler:
                self.gen_lr_scheduler.step()
            if self.disc_lr_scheduler:
                self.disc_lr_scheduler.step()

        return self.save_folder

    def _train_epoch(self, dataloader, criterions, epoch):
        adv_criterion = criterions[0]["loss"]
        recon_criterion1 = criterions[1]["loss"]
        recon_criterion2 = criterions[2]["loss"]
        
        self.gen.train()
        self.disc.train()
        
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
            
            self.disc_optimizer.zero_grad()
            with torch.no_grad():
                fake = self.gen(input)
            disc_fake_hat = self.disc(fake.detach(), input)
            disc_fake_loss = adv_criterion(disc_fake_hat, torch.zeros_like(disc_fake_hat))
            disc_real_hat = self.disc(target, input)
            disc_real_loss = adv_criterion(disc_real_hat, torch.ones_like(disc_real_hat))
            disc_loss = (disc_fake_loss + disc_real_loss) / 2
            
            running_losses[1] += disc_loss.item()
            total_losses[1] += disc_loss.item()
            loss_log[1].append(disc_loss.item())
            
            disc_loss.backward(retain_graph=True)
            self.disc_optimizer.step()
            
            self.gen_optimizer.zero_grad()
            fake_images = self.gen(input)
            eval_fake = self.disc(fake_images, input)
            adv_loss = adv_criterion(eval_fake, torch.ones_like(eval_fake))
            recon_loss = recon_criterion1(fake_images, target)
            recon_loss += recon_criterion2(fake_images, target)
            gen_loss = adv_loss + self.lambda_recon * recon_loss
            
            running_losses[0] += gen_loss.item()
            total_losses[0] += gen_loss.item()
            loss_log[0].append(gen_loss.item())
            
            running_losses[2] += recon_loss.item()
            total_losses[2] += recon_loss.item()
            loss_log[2].append(recon_loss.item())
            
            gen_loss.backward()
            self.gen_optimizer.step()
            
            outer.update(1)
            if (batch_idx + 1) % log_interval == 0:
                
                loss_name = "gen_loss"
                loss_desc = f" Current {loss_name}: {running_losses[0]/log_interval:.5f}"
                running_losses[0] = 0.0
                
                loss_name = "disc_loss"
                loss_desc += (
                    f" Current {loss_name}: {running_losses[1]/log_interval:.5f}"
                )
                running_losses[1] = 0.0
                
                loss_name = "recon_loss"
                loss_desc += (
                    f" Current {loss_name}: {running_losses[2]/log_interval:.5f}"
                )
                running_losses[2] = 0.0
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
        loss_log = [[] for i in range(len(criterions) - 1)]
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
                for i, criterion in enumerate(criterions[1:]):
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
