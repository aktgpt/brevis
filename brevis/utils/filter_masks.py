import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

from skimage.morphology import disk
from skimage.measure import label, regionprops
import os
import tqdm

def plot_multi(cols, rows, images, **kwargs):
    fig, axs = plt.subplots(rows, cols)
    axs = axs.ravel()

    for ax, im in zip(axs, images):
        ax.imshow(im, **kwargs)

    plt.show(block=True)

def normalize_image(image, min_cutoff=None, max_cutoff=None):
    image_norm = np.array((image - min_cutoff) / (max_cutoff - min_cutoff))
    image_norm = np.clip(image_norm, 0, 1)
    return image_norm

def filter_mask(im, mask):

    se = disk(10)

    closed = cv2.morphologyEx(im, cv2.MORPH_CLOSE, se)

    ret2,th2 = cv2.threshold(closed,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    thresh = ((closed > th2)*1).astype('uint8')

    NT = thresh#cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, disk(20))

    labels = label(NT)

    props = regionprops(label_image = labels, intensity_image=closed)

    NT_copy = NT.copy()
    min_area = 1000
    max_area = 10000
    for p in props:
        if p.area < min_area or p.area > max_area:
            NT_copy[p.coords[:,0], p.coords[:,1]] = 0


    # Filter mask

    mask_filt = mask.copy()

    labels = label(mask_filt)
    props = regionprops(label_image = labels, intensity_image=NT_copy)

    for p in props:
        if np.sum(p.intensity_image) == 0:
            mask_filt[p.coords[:,0], p.coords[:,1]] = 0

    return mask_filt

im_folder = '/mnt/hdd2/datasets/astra_data_readonly/astra_data_readonly'
mask_folder = im_folder + '/masks'

folders = ['60x_images']#['20x_images', '40x_images', '60x_images']




for fold in folders: 

    im_files = sorted(glob.glob(im_folder + '/' + fold + '/*'))
    lipid_files = [x for x in im_files if 'C02.tif' in x]

    mask_files = sorted(glob.glob(mask_folder + '/' + fold + '/Nuclei/*'))
    
    save_folder_lipids = im_folder + '/masks_lipids/' + fold + '/Nuclei'
    save_folder_bf = im_folder + '/masks_bf/' + fold + '/Nuclei'

    if not os.path.exists(save_folder_lipids):
        os.makedirs(save_folder_lipids)

    if not os.path.exists(save_folder_bf):
        os.makedirs(save_folder_bf)

    for i in tqdm.tqdm(range(len(mask_files))):
        im = cv2.imread(lipid_files[i], -1)
        mask = cv2.imread(mask_files[i], -1)

        im = (normalize_image(im, np.min(im), np.max(im))*255).astype('uint8')
        mask[mask > 0] = 1

        mask_filt = filter_mask(im, mask)

        cv2.imwrite(save_folder_lipids + '/' + mask_files[i].split('/')[-1], mask_filt*255)
        cv2.imwrite(save_folder_bf + '/' + mask_files[i].split('/')[-1], (mask - mask_filt)*255)

        # plt.imshow(im)
        # plt.imshow(mask_filt, alpha=0.4)
        # plt.show()