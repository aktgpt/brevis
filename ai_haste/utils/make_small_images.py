import numpy as np
import cv2
import glob
import os
import tqdm
im_folder = '/mnt/hdd2/datasets/astra_data_readonly/astra_data_readonly'

folders = ['20x_images', '40x_images', '60x_images']

for fold in folders:

    print(fold)
    save_folder1 = im_folder +  '/' + fold + '_numpy_smaller'

    if not os.path.exists(save_folder1):
        os.makedirs(save_folder1)

    files = sorted(glob.glob(im_folder + '/' + fold + '_numpy/*'))

    for f in tqdm.tqdm(files):
        im = np.load(f)
        if f[-7:-4] == 'C04':
            np.save(save_folder1 + '/' + f.split('/')[-1], im[:1024,:1024, :])
        else:
            np.save(save_folder1 + '/' + f.split('/')[-1], im[:1024,:1024])