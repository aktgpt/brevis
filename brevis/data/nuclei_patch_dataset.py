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

centers = pd.read_csv("exp_stats/nuclei_centers.csv")
centers["brightfield"] = centers["brightfield"].apply(
    lambda x: x.replace(".tiff", ".tif")
)
centers["C1"] = centers["C1"].apply(lambda x: x.replace(".tiff", ".tif"))


class NucleiPatchSegmentDataset(Dataset):
    def __init__(self, config, csv_file, augment=True):
        self.folder = config["folder"]
        self.patch_folder = os.path.join(self.folder, "nuclei_patches")
        self.standardize = config["standardize"]
        self.normalize = config["normalize"]
        self.crop_size = config["crop_size"]
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
        mag_path = magnification + "_images"
        self.getstats(magnification)

        target_name = self.data.iloc[[idx]][self.output_channel].item()
        ## Get input patch of image and create stack of 7 brightfield images
        if self.augmentations:
            r = self.data.iloc[[idx]]["center_r"].item()
            c = self.data.iloc[[idx]]["center_c"].item()
            input_path = (
                os.path.splitext(
                    os.path.join(
                        self.patch_folder,
                        mag_path,
                        self.data.iloc[[idx]]["brightfield"].item(),
                    )
                )[0]
                + f"_r_{r}_c_{c}.npy"
            )
            image = np.load(input_path)
            # if (
            #     not image.shape[0] == self.crop_size[0]
            #     and image.shape[1] == self.crop_size[1]
            # ):
            #     print("shape wrong!!!")
            #     image = np.resize(image, (self.crop_size[0], self.crop_size[1]))

            target_path = (
                os.path.splitext(
                    os.path.join(
                        self.patch_folder,
                        mag_path,
                        self.data.iloc[[idx]][self.output_channel]
                        .item()
                        .replace(".tif", ""),
                    )
                )[0]
                + f"_r_{r}_c_{c}.npy"
            )
            target = np.load(target_path)
            mask_path = os.path.splitext(target_path)[0] + "_mask.npy"
            mask = np.load(mask_path)
        else:
            mag_path = mag_path + "_numpy"
            input_path = (
                os.path.splitext(
                    os.path.join(
                        self.folder,
                        mag_path,
                        self.data.iloc[[idx]]["brightfield"].item(),
                    )
                )[0]
                + ".npy"
            )
            image = np.load(input_path)

            target_path = (
                os.path.splitext(os.path.join(self.folder, mag_path, target_name))[0]
                + ".npy"
            )
            target = np.load(target_path)
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
                    preprocess_stats = [
                        self.fluorecscent_means[channel_idx],
                        self.fluorecscent_stds[channel_idx],
                    ]
                    channel_idx = channel_name_to_idx(self.output_channel)
                    target = (
                        target - self.fluorecscent_means[channel_idx]
                    ) / self.fluorecscent_stds[channel_idx]

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


def get_patch(image, patch_loc):
    img_shape = image.shape
    if patch_loc[0] < 0:
        pad_top = 0 - patch_loc[0]
        x1 = 0
    else:
        pad_top = 0
        x1 = patch_loc[0]
    if patch_loc[1] < 0:
        pad_left = 0 - patch_loc[1]
        y1 = 0
    else:
        pad_left = 0
        y1 = patch_loc[1]
    if patch_loc[2] > img_shape[0] - 1:
        pad_bottom = patch_loc[2] - (img_shape[0] - 1)
        x2 = img_shape[0] - 1
    else:
        pad_bottom = 0
        x2 = patch_loc[2]
    if patch_loc[3] > img_shape[1] - 1:
        pad_right = patch_loc[3] - (img_shape[1] - 1)
        y2 = img_shape[1] - 1
    else:
        pad_right = 0
        y2 = patch_loc[3]

    if len(image.shape) > 2:
        patch = image[x1:x2, y1:y2, :]
        patch = np.pad(
            patch, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), "reflect",
        )
    else:
        patch = image[x1:x2, y1:y2]
        patch = np.pad(patch, ((pad_top, pad_bottom), (pad_left, pad_right)), "reflect")
    return patch
