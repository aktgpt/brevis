from torch.utils.data import Dataset
import numpy as np
import cv2
import pandas as pd
import os
import glob
import math

import albumentations as album

brighfield_means = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]) * 65536
brighfield_stds = np.array([1, 1, 1, 1, 1, 1, 1]) * 65536

fluorecscent_means = np.array([0.5, 0.5, 0.5]) * 65536
fluorecscent_stds = np.array([1, 1, 1]) * 65536

brighfield_maxs = np.array([1, 1, 1, 1, 1, 1, 1]) * 65536
brighfield_mins = np.array([0, 0, 0, 0, 0, 0, 0]) * 65536

fluorecscent_maxs = np.array([1, 1, 1]) * 65536
fluorecscent_mins = np.array([0, 0, 0]) * 65536


class ValidDataset(Dataset):
    def __init__(self, config, csv_file, augment=True):
        self.folder = config["folder"]
        self.standardize = config["standardize"]
        self.normalize = config["normalize"]
        self.data = pd.read_csv(csv_file)

        if config["magnification"] != "all":
            self.data = self.data[
                self.data["magnification"] == config["magnification"]
            ]

        self.output_channel = config["output_channel"]
        self.augment = augment
        if self.augment:
            self.augmentations = load_augmentations(config["augmentations"])
        else:
            self.augmentations = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        ## Get input patch of image and create stack of 7 brightfield images

        input_path = os.path.join(
            self.folder,
            self.data.iloc[[idx]]["magnification"].item() + "_images",
            self.data.iloc[[idx]]["brightfield"].item(),
        )

        image = []
        for Z_nr in range(1, 8):
            image.append(
                cv2.imread(input_path.replace("Z01", "Z0" + str(Z_nr)), -1)
            )

        image = np.array(image).transpose(1, 2, 0).astype("float")

        ## Get output patch of image and create target (either 3 channels or 1 channel)

        if self.output_channel != "all":
            target_path = os.path.join(
                self.folder,
                self.data.iloc[[idx]]["magnification"].item()
                + "_images",
                self.data.iloc[[idx]][self.output_channel].item(),
            )

            target = cv2.imread(target_path, -1).astype("float")

        else:
            target_paths = [
                os.path.join(
                    self.folder,
                    self.data.iloc[[idx]]["magnification"].item()
                    + "_images",
                    self.data.iloc[[idx]]["C1"].item(),
                ),
                os.path.join(
                    self.folder,
                    self.data.iloc[[idx]]["magnification"].item()
                    + "_images",
                    self.data.iloc[[idx]]["C2"].item(),
                ),
                os.path.join(
                    self.folder,
                    self.data.iloc[[idx]]["magnification"].item()
                    + "_images",
                    self.data.iloc[[idx]]["C3"].item(),
                ),
            ]

            target_stack = []

            for target_file in target_paths:
                target_stack.append(cv2.imread(target_file, -1))

            target = np.array(target_stack).transpose(1, 2, 0).astype("float")

        ## Standardization and Normalization

        if self.normalize:
            x = 1
            ### put normalization code here
        else:
            if self.standardize:
                image = (image - brighfield_means) / brighfield_stds

                if self.output_channel != "all":
                    channel_idx = channel_name_to_idx(self.output_channel)
                    target = (
                        target - fluorecscent_means[channel_idx]
                    ) / fluorecscent_stds[channel_idx]

                else:
                    target = (target - fluorecscent_means) / fluorecscent_stds
            else:
                image = image / 65536
                target = target / 65536


        ### Crop evenly spaced tiles

        image = image[0:1024, 0:1024]
        target = target[0:1024, 0:1024]

        tile_size = (512, 512)
        offset = (512, 512)

        image_tiles = []
        target_tiles = []
        for i in range(int(math.ceil(image.shape[0] / (offset[1] * 1.0)))):
            for j in range(int(math.ceil(image.shape[1] / (offset[0] * 1.0)))):
                image_crop = image[offset[1] * i:min(offset[1] * i + tile_size[1], image.shape[0]),
                            offset[0] * j:min(offset[0] * j + tile_size[0], image.shape[1])]
                
                if image_crop.shape[0] == 512 and image_crop.shape[1] == 512:
                    image_tiles.append(image_crop)
                    
                    target_tiles.append(target[offset[1] * i:min(offset[1] * i + tile_size[1], image.shape[0]),
                                offset[0] * j:min(offset[0] * j + tile_size[0], image.shape[1])])

        image_tiles = np.array(image_tiles)
        target_tiles = np.array(target_tiles)
        ## Augmentations

        # if self.augmentations:
        #     augmented = self.augmentations(image=image, mask=target)
        #     image = augmented["image"]
        #     target = augmented["mask"]

        image = image_tiles.transpose(0, 3, 1, 2).astype(np.float32)
        if len(target_tiles.shape) > 3:
            target = target_tiles.transpose(0, 3, 1, 2).astype(np.float32)
        else:
            target = target_tiles
        return image, target


def standardize_image(channel1, min_cutoff=None, max_cutoff=None):
    if min_cutoff is None:
        min_cutoff = np.min(channel1)
    if max_cutoff is None:
        max_cutoff = np.max(channel1)
    if max_cutoff == 0.0 and min_cutoff == 0.0:
        return channel1
    else:
        channel = channel1.copy()
        channel = channel.astype(np.float32)
        channel[channel < min_cutoff] = min_cutoff
        channel[channel > max_cutoff] = max_cutoff
        channel = channel - min_cutoff
        channel = (channel) / (max_cutoff - min_cutoff)
        channel = np.clip(channel, 0, 1)
        return channel


def linear_normalization_to_8bit(im, min_value, max_value):
    im_norm = np.round(
        np.clip(
            ((im.astype("float") - min_value) / (max_value - min_value)) * 255,
            0,
            255,
        )
    ).astype("uint8")
    return im_norm


def channel_name_to_idx(channel_name):
    if channel_name == "C1":
        return 0
    elif channel_name == "C2":
        return 1
    elif channel_name == "C3":
        return 2
    else:
        print("Channel name not defined")
        return None


def load_augmentations(augentations):
    augs = []

    for augment_type in augentations:

        augs.append(
            getattr(album, augment_type["type"])(**augment_type["args"])
        )

    return album.Compose(augs)
