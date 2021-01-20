import os

import cv2
import numpy as np
from .read_images import AdipocyteDataProcessor
import pandas as pd


class ExperimentStatsGetter:
    def __init__(self, data_folder, save_folder):
        self.magnifications = ("20x", "40x", "60x")
        self.mag_folders = [
            os.path.join(data_folder, f"{mag}_images") for mag in self.magnifications
        ]
        self.mag_images = []
        self.channel_maps = []
        for i, mag_folder in enumerate(self.mag_folders):
            folder_processor = AdipocyteDataProcessor(mag_folder)
            well_images, channel_map = folder_processor.group_channels(groupby_prop=("well", "F"))
            self.mag_images.append(well_images)
            self.channel_maps.append(channel_map)
        self.save_folder = save_folder

    def get_stats(self):
        for i, well_images in enumerate(self.mag_images):
            channel_map = self.channel_maps[i]
            n_channels = len(well_images[0])
            n_images = len(well_images)

            all_well_images = [[[] for i in range(n_channels)] for i in range(n_images)]
            for j, well_id in enumerate(well_images):
                for k, channel_name in enumerate(well_id):
                    channel_idx = np.where(
                        (np.array(channel_map[1]) == channel_name["C"])
                        & (np.array(channel_map[2]) == channel_name["Z"])
                    )[0][0]
                    channel = cv2.imread(
                        os.path.join(channel_name["folder"], channel_name["imagename"]), -1,
                    )
                    all_well_images[j][channel_idx] = channel

            all_well_images = np.array(all_well_images)
            mean = np.mean(all_well_images, axis=(0, 2, 3))
            var = np.std(all_well_images, axis=(0, 2, 3))
            min = np.min(all_well_images, axis=(0, 2, 3))
            max = np.max(all_well_images, axis=(0, 2, 3))
            max_per = np.percentile(all_well_images, q=99.99, axis=(0, 2, 3))
            min_per = np.percentile(all_well_images, q=0.01, axis=(0, 2, 3))
            df = pd.DataFrame(
                {
                    "C": channel_map[1],
                    "Z": channel_map[2],
                    "mean": mean,
                    "var": var,
                    "min": min,
                    "max": max,
                    "min_per": min_per,
                    "max_per": max_per,
                }
            )
            df.to_csv(
                os.path.join(self.save_folder, f"{self.magnifications[i]}_stats.csv"), index=False
            )


