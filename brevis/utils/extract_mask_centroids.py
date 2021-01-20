import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

from skimage.morphology import disk
from skimage.measure import label, regionprops
import os 
import tqdm


im_folder = "/mnt/hdd2/datasets/astra_data_readonly/astra_data_readonly"

folders = ["60x_images"]

for fold in folders:

    lipid_mask_folder = im_folder + "/masks_lipids/" + fold + "/Nuclei"
    bf_mask_folder = im_folder + "/masks_bf/" + fold + "/Nuclei"

    save_folder_lipids = im_folder + "/masks_lipids_centroids/" + fold + "/Nuclei"
    save_folder_bf = im_folder + "/masks_bf_centroids/" + fold + "/Nuclei"

    if not os.path.exists(save_folder_lipids): 
        os.makedirs(save_folder_lipids)

    if not os.path.exists(save_folder_bf):
        os.makedirs(save_folder_bf)

    lipid_mask_files = sorted(glob.glob(lipid_mask_folder + "/*"))
    bf_mask_files = sorted(glob.glob(bf_mask_folder + "/*"))

    for i in tqdm.tqdm(range(len(lipid_mask_files))):
        lipid_mask = cv2.imread(lipid_mask_files[i], -1)
        bf_mask = cv2.imread(bf_mask_files[i], -1)

        lipid_mask[lipid_mask > 0] = 1
        bf_mask[bf_mask > 0] = 1

        lipid_labels = label(lipid_mask)
        lipid_props = regionprops(lipid_labels)

        bf_labels = label(bf_mask)
        bf_props = regionprops(bf_labels)

        lipid_centroids = []

        for p in lipid_props:
            r, c = p.centroid
            lipid_centroids.append([int(r), int(c)])

        lipid_centroids = np.array(lipid_centroids)

        np.save(
            save_folder_lipids
            + "/"
            + lipid_mask_files[i].split("/")[-1].replace(".tiff", ".npy"),
            lipid_centroids,
        )

        bf_centroids = []

        for p in bf_props:
            r, c = p.centroid
            bf_centroids.append([int(r), int(c)])

        bf_centroids = np.array(bf_centroids)

        np.save(
            save_folder_bf
            + "/"
            + bf_mask_files[i].split("/")[-1].replace(".tiff", ".npy"),
            bf_centroids,
        )
