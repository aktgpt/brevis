import glob
import os
import random
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import albumentations as album
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

stats_20x = pd.read_csv("exp_stats/20x_stats.csv")
stats_40x = pd.read_csv("exp_stats/40x_stats.csv")
stats_60x = pd.read_csv("exp_stats/60x_stats.csv")

brighfield_means_20x = np.array(stats_20x.iloc[3:]["mean"])
fluorecscent_means_20x = np.array(stats_20x.iloc[:3]["mean"])
brighfield_means_40x = np.array(stats_40x.iloc[3:]["mean"])
fluorecscent_means_40x = np.array(stats_40x.iloc[:3]["mean"])
brighfield_means_60x = np.array(stats_60x.iloc[3:]["mean"])
fluorecscent_means_60x = np.array(stats_60x.iloc[:3]["mean"])

centers = pd.read_csv("exp_stats/nuclei_centers.csv")
centers["brightfield"] = centers["brightfield"].apply(
    lambda x: x.replace(".tiff", ".tif")
)
centers["C1"] = centers["C1"].apply(lambda x: x.replace(".tiff", ".tif"))


class RandomCenterCropDataset(Dataset):
    def __init__(self, config, csv_file, augment=True):
        self.folder = config["folder"]
        self.standardize = config["standardize"]
        self.normalize = config["normalize"]
        self.crop_size = config["crop_size"]
        self.maximum_offset = self.crop_size[0] // 2
        self.p = 0.8
        data = pd.read_csv(csv_file)

        if augment:
            self.data = centers[
                centers["brightfield"].isin(list(data["brightfield"]))
            ].reset_index(drop=True)
        else:
            self.data = data

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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        magnification = self.data.iloc[[idx]]["magnification"].item()
        mag_path = magnification + "_images_numpy"
        self.getstats(magnification)

        ## Get input patch of image and create stack of 7 brightfield images
        # input_path = (
        #     os.path.splitext(
        #         os.path.join(
        #             self.folder, mag_path, self.data.iloc[[idx]]["brightfield"].item(),
        #         )
        #     )[0]
        #     + ".npy"
        # )

        # image = np.load(input_path)

        input_path = os.path.join(
            self.folder, mag_path, self.data.iloc[[idx]]["brightfield"].item(),
        )
        image = []
        for Z_nr in range(1, 8):
            image.append(cv2.imread(input_path.replace("Z01", "Z0" + str(Z_nr)), -1))

        image = np.array(image).transpose(1, 2, 0).astype("float")

        ## Get output patch of image and create target (either 3 channels or 1 channel)
        if self.output_channel != "all":
            # target_name = self.data.iloc[[idx]][self.output_channel].item()
            # target_path = (
            #     os.path.splitext(os.path.join(self.folder, mag_path, target_name))[0]
            #     + ".npy"
            # )
            # target = np.load(target_path)
            target_name = self.data.iloc[[idx]][self.output_channel].item()
            target_path = os.path.join(self.folder, mag_path, target_name,)
            target = cv2.imread(target_path, -1).astype("float")

            mask_mag_path = magnification + "_images"
            mask_path = os.path.join(
                self.folder,
                "masks",
                mask_mag_path,
                "Nuclei" if "C1" in self.output_channel else "Cytoplasm",
                self.data.iloc[[idx]][self.output_channel]
                .item()
                .replace(".tif", ".tiff"),
            )
            mask = cv2.imread(mask_path, -1)
            if self.output_channel == "C1":
                mask[mask > 0] = 1

        ## Random crop around nuclei
        width, height = target.shape
        if self.augmentations:
            if random.random() < self.p:
                r = self.data.iloc[[idx]]["center_r"].values.item()
                c = self.data.iloc[[idx]]["center_c"].values.item()
            else:
                zero_yx = np.argwhere(mask == 0)
                r, c = random.choice(zero_yx)
                
            xmin = (r - self.crop_size[0] // 2) - random.randint(
                -self.maximum_offset, self.maximum_offset
            )
            ymin = (c - self.crop_size[1] // 2) - random.randint(
                -self.maximum_offset, self.maximum_offset
            )

            xmin = np.clip(xmin, 0, width - self.crop_size[0])
            ymin = np.clip(ymin, 0, height - self.crop_size[1])

            xmax = xmin + self.crop_size[0]
            ymax = ymin + self.crop_size[1]

            image = crop(image, xmin, ymin, xmax, ymax)
            target = crop(target, xmin, ymin, xmax, ymax)
            mask = crop(mask, xmin, ymin, xmax, ymax)

        ## Standardization and Normalization

        if self.normalize:
            image = normalize_image(image, self.brightfield_min, self.brightfield_max)
            preprocess_step = "normalize"

            if self.output_channel != "all":
                channel_idx = channel_name_to_idx(self.output_channel)
                preprocess_stats = [
                    self.fluorecscent_min[channel_idx],
                    self.fluorecscent_max[channel_idx],
                ]
                target = normalize_image(
                    target,
                    self.fluorecscent_min[channel_idx],
                    self.fluorecscent_max[channel_idx],
                )
        else:
            if self.standardize:
                preprocess_step = "standardize"
                image = (image - self.brightfield_means) / self.brightfield_stds
                if self.output_channel != "all":
                    channel_idx = channel_name_to_idx(self.output_channel)
                    preprocess_stats = [
                        self.fluorecscent_means[channel_idx],
                        self.fluorecscent_stds[channel_idx],
                    ]
                    target = (
                        target - self.fluorecscent_means[channel_idx]
                    ) / self.fluorecscent_stds[channel_idx]
            else:
                image = image / 65536
                target = target / 65536

        ## Augmentations
        target = np.dstack((mask, target))

        if self.augmentations:
            augmented = self.augmentations(image=image, mask=target)
            image = augmented["image"]
            target = augmented["mask"]

        image = image.transpose(2, 0, 1).astype(np.float32)
        if len(target.shape) > 2:
            target = target.transpose(2, 0, 1).astype(np.float32)
        else:
            target = target[np.newaxis, ...].astype(np.float32)
        return (
            image,
            target,
            target_name,
            preprocess_step,
            preprocess_stats,
            magnification,
        )

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
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                height=height,
                width=width,
            )
        )

    return img[x_min:x_max, y_min:y_max]
