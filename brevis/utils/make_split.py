from .read_images import YokogawaDataProcessor
import pandas as pd
import operator
from itertools import groupby
import os


class DataSplitter:
    def __init__(self, data_folder, save_folder):
        self.magnifications = ("20x", "40x", "60x")
        self.mag_folders = [
            os.path.join(data_folder, f"{mag}_images") for mag in self.magnifications
        ]
        self.save_folder = save_folder

    def make_split(self):
        n_splits = 8
        train_splits = [[] for i in range(n_splits)]
        test_splits = [[] for i in range(n_splits)]
        for i, mag_folder in enumerate(self.mag_folders):
            magnification = self.magnifications[i]
            folder_processor = YokogawaDataProcessor(mag_folder)

            for i in range(n_splits):
                well_images, _ = folder_processor.group_channels()
                test_well = well_images.pop(i)
                train_well = [item for sublist in well_images for item in sublist]

                grouped_test = groupby_fov(test_well)
                grouped_train = groupby_fov(train_well)

                df_train_split = make_df(grouped_train, magnification)
                df_test_split = make_df(grouped_test, magnification)

                train_splits[i].append(df_train_split)
                test_splits[i].append(df_test_split)

        for i in range(n_splits):
            train_df = pd.concat(train_splits[i]).reset_index(drop=True)
            test_df = pd.concat(test_splits[i])
            train_df.to_csv(os.path.join(self.save_folder, f"train_split_{i+1}.csv"), index=False)
            test_df.to_csv(os.path.join(self.save_folder, f"test_split_{i+1}.csv"), index=False)


def groupby_fov(images):
    grouper = operator.itemgetter("well", "F")
    images.sort(key=grouper)
    grouped_images = []
    for _, v in groupby(images, key=grouper):
        grouped_images.append(list(v))
    return grouped_images


def make_df(grouped_images, magnification):
    magnifications = []
    folders = []
    brightfield = []
    c1 = []
    c2 = []
    c3 = []
    for fov in grouped_images:
        magnifications.append(magnification)
        folders.append(fov[0]["folder"])
        for image in fov:
            if image["C"] == "01":
                c1.append(image["imagename"])
            elif image["C"] == "02":
                c2.append(image["imagename"])
            elif image["C"] == "03":
                c3.append(image["imagename"])

            elif image["C"] == "04" and image["Z"] == "01":
                brightfield.append(image["imagename"])

    df = pd.DataFrame(
        {
            "magnification": magnifications,
            "folder": folders,
            "brightfield": brightfield,
            "C1": c1,
            "C2": c2,
            "C3": c3,
        }
    )
    return df
