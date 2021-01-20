import os

import cv2
import numpy as np
from glob import glob
import pandas as pd
from skimage.measure import label, regionprops
from tqdm import tqdm

data_folder = "/workspace/astra_data"
magnifications = ("20x", "40x", "60x")
crop_size = [512, 512]


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


def main():
    mag_folders = [
        os.path.join(data_folder, "masks", f"{mag}_images", "Nuclei")
        for mag in magnifications
    ]
    mag_images = []
    channel_maps = []
    for i, mag_folder in enumerate(mag_folders):
        mask_images = sorted(glob(mag_folder + "/*"))
        mag_images.append(mask_images)

    all_nuclei_images = []
    all_bf_images = []
    all_magnification = []

    save_folder = os.path.join(data_folder, "nuclei_patches")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for i, mask_images in enumerate(mag_images):
        mag_level = magnifications[i]
        bf_folder = os.path.join(data_folder, f"{mag_level}_images_numpy")
        mag_save_folder = os.path.join(save_folder, f"{mag_level}_images")
        if not os.path.exists(mag_save_folder):
            os.makedirs(mag_save_folder)

        for j, mask_image in enumerate(tqdm(mask_images)):
            mask = (cv2.imread(mask_image, -1) > 0).astype(np.uint8)
            image_name = os.path.basename(mask_image).replace(".tiff", ".tif")
            bf_image_name = image_name.replace("C01", "C04").replace("A01", "A04")
            bf_image = np.load(
                os.path.join(bf_folder, bf_image_name.replace(".tif", ".npy"))
            )
            fl_image = np.load(
                os.path.join(bf_folder, image_name.replace(".tif", ".npy"))
            )
            labels = label(mask)
            props = regionprops(labels)
            centroids = []
            for p in props:
                r, c = p.centroid
                patch_loc = [
                    int(int(r) - crop_size[0] / 2),
                    int(int(c) - crop_size[1] / 2),
                    int(int(r) + crop_size[0] / 2),
                    int(int(c) + crop_size[1] / 2),
                ]
                mask_patch = get_patch(mask, patch_loc)
                bf_patch = get_patch(bf_image, patch_loc)
                fl_patch = get_patch(fl_image, patch_loc)
                if (
                    (mask_patch.shape[0] != crop_size[0])
                    and (mask_patch.shape[1] != crop_size[1])
                ):
                    print(mask_patch.shape)
                np.save(
                    os.path.join(
                        mag_save_folder,
                        f"{os.path.splitext(image_name)[0]}_r_{int(r)}_c_{int(c)}",
                    ),
                    fl_patch,
                )
                np.save(
                    os.path.join(
                        mag_save_folder,
                        f"{os.path.splitext(image_name)[0]}_r_{int(r)}_c_{int(c)}_mask",
                    ),
                    mask_patch,
                )
                np.save(
                    os.path.join(
                        mag_save_folder,
                        f"{os.path.splitext(bf_image_name)[0]}_r_{int(r)}_c_{int(c)}",
                    ),
                    bf_patch,
                )


main()
