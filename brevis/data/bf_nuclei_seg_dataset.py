from torch.utils.data import Dataset
import numpy as np
import cv2
import pandas as pd
import os
import glob
import random

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

def check_bounds_of_crop(c, crop_size, im_size):
    half_size = crop_size // 2
    if c[0]- half_size < 0 or c[0] + half_size > im_size[0]:
        return True
    elif c[1] - half_size < 0 or c[1] + half_size > im_size[1]:
        return True
    else:
        return False

def crop(img, x_min, y_min, x_max, y_max):
    height, width = img.shape[:2]
    if x_max <= x_min or y_max <= y_min:
        raise ValueError(
            "We should have x_min < x_max and y_min < y_max. But we got"
            " (x_min = {x_min}, y_min = {y_min}, x_max = {x_max}, y_max = {y_max})".format(
                x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max
            )
        )

    if x_min < 0 or x_max > height or y_min < 0 or y_max > width:
        raise ValueError(
            "Values for crop should be non negative and equal or smaller than image sizes"
            "(x_min = {x_min}, y_min = {y_min}, x_max = {x_max}, y_max = {y_max}, "
            "height = {height}, width = {width})".format(
                x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, height=height, width=width
            )
        )

    return img[x_min:x_max, y_min:y_max]


class BFNucSegDataset(Dataset):
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

        self.safe_crop = config["safe_crop"]
        self.crop_size = config["crop_size"]
        self.p = config["p"]

        if self.safe_crop:
            self.coord_folder = os.path.join(config["folder"], 'masks_bf_centroids', 
            config["magnification"] + '_images', 'Nuclei')

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

        # if self.use_npy:
        #     image = np.load(input_path.replace(mag_path, mag_path + '_numpy' + 
        #     ('_smaller' if self.use_smaller_dataset else '')).replace('.tif', '.npy'))
        # else:
        #     image = []
        #     for Z_nr in range(1, 8):
        #         image.append(cv2.imread(input_path.replace("Z01", "Z0" + str(Z_nr)), -1))
                
        #     image = np.array(image).transpose(1, 2, 0).astype("float")

        # if self.use_npy:

        #     image = np.load(os.path.join(
        #         self.folder, mag_path + '_numpy' + 
        #         ('_smaller' if self.use_smaller_dataset else ''), self.data.iloc[[idx]]['C2'].item(),
        #     ).replace('.tif', '.npy'))

        # else:
        #     image = cv2.imread(os.path.join(
        #         self.folder, mag_path, self.data.iloc[[idx]]['C2'].item(),
        #     ), -1)

        image = []
        for Z_nr in range(1, 8):
            image.append(cv2.imread(input_path.replace("Z01", "Z0" + str(Z_nr)), -1))

        image = np.array(image).transpose(1, 2, 0).astype("float")

        ## Get output patch of image and create target (either 3 channels or 1 channel)

        target_name = self.data.iloc[[idx]][self.output_channel].item()
        target_path = os.path.join(
            self.folder, 'masks_bf', mag_path, 
            'Nuclei', self.data.iloc[[idx]][self.output_channel].item().replace('.tif','.tiff')
        )

        target = cv2.imread(target_path, -1).astype("float")
        if self.use_smaller_dataset:
            target = target[:1024, :1024]
        if self.output_channel == 'C1':
            target[target > 0] = 1

        ## Standardization and Normalization

        if self.normalize:
            preprocess_step = "normalize"
            image = normalize_image(image, self.brightfield_min, self.brightfield_max)
        else:
            image = image / 65536


        maximum_offset = self.crop_size // 2
        width, height = target.shape        
        
        ## Safe crop
        if self.augmentations:
            if self.safe_crop:

                if random.random() < self.p:
                    crop_coords = np.load(os.path.join(self.coord_folder, 
                                    self.data.iloc[[idx]][self.output_channel].item().replace('.tif','.npy')
                                    ))
                    coord_idx = np.random.randint(len(crop_coords))
                    c = crop_coords[coord_idx]
                else:
                    zero_yx = np.argwhere(target == 0)     
                    x, y = random.choice(zero_yx)
                    c = [x,y]

                xmin = (c[0] - self.crop_size // 2) - random.randint(-maximum_offset, maximum_offset)
                ymin = (c[1] - self.crop_size // 2) - random.randint(-maximum_offset, maximum_offset)

                xmin = np.clip(xmin, 0, width - self.crop_size)
                ymin = np.clip(ymin, 0, height - self.crop_size)

                xmax = xmin + self.crop_size
                ymax = ymin + self.crop_size

                image = crop(image, xmin, ymin, xmax, ymax)
                target = crop(target, xmin, ymin, xmax, ymax)

            ## Albumentations
            augmented = self.augmentations(image=image, mask=target)
            image = augmented["image"]
            target = augmented["mask"]

        if len(image.shape) > 2:
            image = image.transpose(2, 0, 1).astype(np.float32)
        else: 
            image = image[np.newaxis, ...].astype(np.float32)

        # if len(target.shape) > 2:
        #     target = target.transpose(2, 0, 1).astype(np.float32)
        # else:
        #     target = target[np.newaxis, ...].astype(np.float32)

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
        np.clip(
            ((im.astype("float") - min_value) / (max_value - min_value)) * 255, 0, 255,
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

        augs.append(getattr(album, augment_type["type"])(**augment_type["args"]))

    return album.Compose(augs)
