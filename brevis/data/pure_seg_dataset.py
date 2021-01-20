from torch.utils.data import Dataset
import numpy as np
import cv2
import pandas as pd
import os
import glob

import albumentations as album


stats_20x = pd.read_csv("exp_stats/20x_stats.csv")
stats_40x = pd.read_csv("exp_stats/40x_stats.csv")
stats_60x = pd.read_csv("exp_stats/60x_stats.csv")

brighfield_means_20x = np.array(stats_20x.iloc[3:]["mean"])
fluorecscent_means_20x = np.array(stats_20x.iloc[:3]["mean"])
brighfield_means_40x = np.array(stats_40x.iloc[3:]["mean"])
fluorecscent_means_40x = np.array(stats_40x.iloc[:3]["mean"])
brighfield_means_60x = np.array(stats_60x.iloc[3:]["mean"])
fluorecscent_means_60x = np.array(stats_60x.iloc[:3]["mean"])


class PureSegDataset(Dataset):
    def __init__(self, config, csv_file, augment=True):
        self.folder = config["folder"]
        self.standardize = config["standardize"]
        self.normalize = config["normalize"]
        self.data = pd.read_csv(csv_file)

        if config["magnification"] != "all":
            self.data = self.data[
                self.data["magnification"] == config["magnification"]
            ].reset_index(drop=True)

        self.output_channel = config["output_channel"]
        self.augment = augment
        if self.augment:
            self.augmentations = load_augmentations(config["augmentations"])
        else:
            self.augmentations = None

        self.use_npy = config['use_npy']
        self.use_fluorescent = config['use_fluorescent']
        self.use_smaller_dataset = config["use_smaller_dataset"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        magnification = self.data.iloc[[idx]]["magnification"].item()
        mag_path = magnification + "_images"
        self.getstats(magnification)

        ## Get input patch of image and create stack of 7 brightfield images

        input_path = os.path.join(
            self.folder, mag_path, self.data.iloc[[idx]]["brightfield"].item(),
        )

        if self.use_npy:
            image = np.load(input_path.replace(mag_path, mag_path + '_numpy' + 
            ('_smaller' if self.use_smaller_dataset else '')).replace('.tif', '.npy'))
        else:
            image = []
            for Z_nr in range(1, 8):
                image.append(cv2.imread(input_path.replace("Z01", "Z0" + str(Z_nr)), -1))
                
            image = np.array(image).transpose(1, 2, 0).astype("float")

        if self.use_fluorescent:
            other_channels = ['C1', 'C2', 'C3']
            other_channels.remove(self.output_channel)
            if self.use_npy:

                extra_channel1 = np.load(os.path.join(
                    self.folder, mag_path + '_numpy' + 
                    ('_smaller' if self.use_smaller_dataset else ''), self.data.iloc[[idx]][other_channels[0]].item(),
                ).replace('.tif', '.npy'))

                extra_channel2 = np.load(os.path.join(
                    self.folder, mag_path + '_numpy' + 
                    ('_smaller' if self.use_smaller_dataset else ''), self.data.iloc[[idx]][other_channels[1]].item(),
                ).replace('.tif', '.npy'))      

            else:
                extra_channel1 = cv2.imread(os.path.join(
                    self.folder, mag_path, self.data.iloc[[idx]][other_channels[0]].item(),
                ), -1)

                extra_channel2 = cv2.imread(os.path.join(
                    self.folder, mag_path, self.data.iloc[[idx]][other_channels[1]].item(),
                ), -1)
        ## Get output patch of image and create target (either 3 channels or 1 channel)
        target_name = self.data.iloc[[idx]][self.output_channel].item()
        target_path = os.path.join(
            self.folder, 'masks', mag_path, 
            'Nuclei' if 'C1' in self.output_channel else 'Cytoplasm', 
            self.data.iloc[[idx]][self.output_channel].item().replace('.tif','.tiff')
        )

        target = cv2.imread(target_path, -1).astype("float")
        if self.use_smaller_dataset:
            target = target[:1024, :1024]
        if self.output_channel == 'C1':
            target[target > 0] = 1

        ## Standardization and Normalization

        if self.normalize:
            image = normalize_image(image, self.brightfield_min, self.brightfield_max)
            preprocess_step = "normalize"
            
            if self.use_fluorescent:
                extra_channel1 = normalize_image(
                    extra_channel1, self.fluorecscent_min[channel_name_to_idx(other_channels[0])], self.fluorecscent_max[channel_name_to_idx(other_channels[0])]
                )

                extra_channel2 = normalize_image(
                    extra_channel2, self.fluorecscent_min[channel_name_to_idx(other_channels[1])], self.fluorecscent_max[channel_name_to_idx(other_channels[1])]
                )

        else:
            if self.standardize:
                image = (image - self.brightfield_means) / self.brightfield_stds
            else:
                image = image / 65536

        ## Augmentations
        
        if self.use_fluorescent:
            image = np.dstack((np.max(image, 2), extra_channel1, extra_channel2))

        if self.augmentations:
            augmented = self.augmentations(image=image, mask=target)
            image = augmented["image"]
            target = augmented["mask"]

        image = image.transpose(2, 0, 1).astype(np.float32)

        return image, target, target_name

    def getstats(self, magnification):
        if magnification == "20x":
            self.brightfield_means = np.array(stats_20x.iloc[3:]["mean"])
            self.fluorecscent_means = np.array(stats_20x.iloc[:3]["mean"])
            self.brightfield_stds = np.array(stats_20x.iloc[3:]["var"])
            self.fluorecscent_stds = np.array(stats_20x.iloc[:3]["var"])
            self.brightfield_max = np.array(stats_20x.iloc[3:]["max"])
            self.fluorecscent_max = np.array(stats_20x.iloc[:3]["max"])
            self.brightfield_min = np.array(stats_20x.iloc[3:]["min"])
            self.fluorecscent_min = np.array(stats_20x.iloc[:3]["min"])

        elif magnification == "40x":
            self.brightfield_means = np.array(stats_40x.iloc[3:]["mean"])
            self.fluorecscent_means = np.array(stats_40x.iloc[:3]["mean"])
            self.brightfield_stds = np.array(stats_40x.iloc[3:]["var"])
            self.fluorecscent_stds = np.array(stats_40x.iloc[:3]["var"])
            self.brightfield_max = np.array(stats_40x.iloc[3:]["max"])
            self.fluorecscent_max = np.array(stats_40x.iloc[:3]["max"])
            self.brightfield_min = np.array(stats_40x.iloc[3:]["min"])
            self.fluorecscent_min = np.array(stats_40x.iloc[:3]["min"])

        elif magnification == "60x":
            self.brightfield_means = np.array(stats_60x.iloc[3:]["mean"])
            self.fluorecscent_means = np.array(stats_60x.iloc[:3]["mean"])
            self.brightfield_stds = np.array(stats_60x.iloc[3:]["var"])
            self.fluorecscent_stds = np.array(stats_60x.iloc[:3]["var"])
            self.brightfield_max = np.array(stats_60x.iloc[3:]["max"])
            self.fluorecscent_max = np.array(stats_60x.iloc[:3]["max"])
            self.brightfield_min = np.array(stats_60x.iloc[3:]["min"])
            self.fluorecscent_min = np.array(stats_60x.iloc[:3]["min"])


def normalize_image(image, min_cutoff=None, max_cutoff=None):
    image_norm = np.array((image - min_cutoff) / (max_cutoff - min_cutoff))
    image_norm = np.clip(image_norm, 0, 1)
    return image_norm


def linear_normalization_to_8bit(im, min_value, max_value):
    im_norm = np.round(
        np.clip(((im.astype("float") - min_value) / (max_value - min_value)) * 255, 0, 255,)
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

        augs.append(getattr(album, augment_type["type"])(**augment_type["args"]))

    return album.Compose(augs)
