import os

import cv2
import numpy as np
from glob import glob
import pandas as pd
from skimage.measure import label, regionprops


class CentroidExtractor:
    def __init__(self, data_folder, save_folder):
        self.magnifications = ("20x", "40x", "60x")
        self.mag_folders = [
            os.path.join(data_folder, "masks", f"{mag}_images", "Nuclei")
            for mag in self.magnifications
        ]
        self.mag_images = []
        self.channel_maps = []
        for i, mag_folder in enumerate(self.mag_folders):
            mask_images = sorted(glob(mag_folder + "/*"))
            self.mag_images.append(mask_images)
        self.save_folder = save_folder

    def get_centroids(self):
        all_nuclei_images = []
        all_bf_images = []
        all_center_r = []
        all_center_c = []
        all_magnification = []
        for i, mask_images in enumerate(self.mag_images):
            mag_level = self.magnifications[i]
            for j, mask_image in enumerate(mask_images):
                mask = (cv2.imread(mask_image, -1) > 0).astype(np.uint8)
                image_name = os.path.basename(mask_image).replace(".tiff", ".tif")
                bf_image_name = image_name.replace("C01", "C04").replace("A01", "A04")
                labels = label(mask)
                props = regionprops(labels)
                centroids = []
                for p in props:
                    r, c = p.centroid
                    all_magnification.append(mag_level)
                    all_nuclei_images.append(image_name)
                    all_bf_images.append(bf_image_name)
                    all_center_c.append(int(c))
                    all_center_r.append(int(r))
        df = pd.DataFrame(
            {
                "magnification": all_magnification,
                "brightfield": all_bf_images,
                "C1": all_nuclei_images,
                "center_r": all_center_r,
                "center_c": all_center_c,
            }
        )
        df.to_csv(os.path.join(self.save_folder, "nuclei_centers.csv"), index=False)
