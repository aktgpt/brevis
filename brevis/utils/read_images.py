from glob import glob
from os import walk, path
import operator
from itertools import groupby


def read_multiple_datasets(folder):
    data_folders = []
    types = ("/*.tif", "/*.tiff", "/*.BMP")
    for dirpath, _, _ in walk(folder):
        image_wfolder_list = []
        for files in types:
            image_wfolder_list.extend(glob(dirpath + files))
        if len(image_wfolder_list) > 5:
            data_folders.append(dirpath)
    data_folders.sort()
    return data_folders


def get_images_from_folder(image_source):
    # depending on the modality
    types = ("/*.tif", "/*.tiff", "/*.BMP")
    image_wfolder_list = []
    for files in types:
        image_wfolder_list.extend(glob(image_source + files))
    image_list = []
    for i in range(len(image_wfolder_list)):
        _, filename = path.split(image_wfolder_list[i])
        image_list.append(filename)
    return image_list


class AdipocyteDataProcessor:
    def __init__(self, image_source):
        self.image_source = image_source
        # data_folders = read_multiple_datasets(image_source)[:3]
        self.images = get_images_from_folder(image_source)
        # for folder in data_folders:
        #     self.images.extend()

    def get_prop_dict(self):
        self.prop_dict = dict(
            folder="%s",
            imagename="%s",
            plateid="%s",
            well="{:1s}{:2d}",
            T="{:4d}",
            F="{:3d}",
            L="{:2d}",
            A="{:2d}",
            Z="{:2d}",
            C="{:2d}",
        )

    def extract_image_props(self, image_name):
        self.get_prop_dict()
        image_props = self.prop_dict
        plateid_well, _, props = image_name.rpartition("_")
        plateid, _, well = plateid_well.rpartition("_")
        image_props["folder"] = self.image_source
        image_props["imagename"] = image_name
        image_props["plateid"] = plateid
        image_props["well"] = well
        image_props["T"] = props[props.find("T") + 1 : props.find("T") + 5]
        image_props["F"] = props[props.find("F") + 1 : props.find("F") + 4]
        image_props["L"] = props[props.find("L") + 1 : props.find("L") + 3]
        image_props["A"] = props[props.find("A") + 1 : props.find("A") + 3]
        image_props["Z"] = props[props.find("Z") + 1 : props.find("Z") + 3]
        image_props["C"] = props[props.find("C") + 1 : props.find("C") + 3]
        return image_props

    def group_channels(self, groupby_prop=("well",)):
        prop_list = []
        for i in range(len(self.images)):
            img_props = self.extract_image_props(self.images[i])
            if not img_props["well"][0].isalpha() or not img_props["C"].isdigit():
                continue
            else:
                prop_list.append(img_props)
        grouper = operator.itemgetter(*groupby_prop)
        prop_list.sort(key=grouper)
        grouped_images = []
        for _, v in groupby(prop_list, key=grouper):
            grouped_images.append(list(v))
        # self.well_images = grouped_images
        return grouped_images, self.get_channel_map(grouped_images)

    def get_channel_map(self, well_images):
        channels = list(range(len(well_images[0])))
        channel_idx = sorted([well_images[0][i]["C"] for i in range(len(well_images[0]))])
        z_idx = sorted([well_images[0][i]["Z"] for i in range(len(well_images[0]))])
        # check if duplicates
        # if len(channel_idx) != len(set(channel_idx)):
        #     print("something is fishy")
        return channels, channel_idx, z_idx
