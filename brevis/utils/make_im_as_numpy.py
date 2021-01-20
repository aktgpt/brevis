import numpy as np
import cv2
import glob
import os

im_folder = 'adipocyte_data'

folders = ['20x_images', '40x_images', '60x_images']

for fold in folders:

    print(fold)
    save_folder1 = im_folder +  '/' + fold + '_numpy'
    save_folder2 = im_folder.replace('hdd2','hdd1') + '/' + fold + '_numpy'

    if not os.path.exists(save_folder1):
        os.makedirs(save_folder1)
    
    if not os.path.exists(save_folder2):
        os.makedirs(save_folder2)

    files = sorted(glob.glob(im_folder + '/' + fold + '/*'))

    i = 0
    while i < len(files):
        f = files[i]
        channel = f[-7:-4]
        
        if channel != 'C04':
            #print(f.split('/')[-1])
            name = files[i].split('/')[-1].replace('.tif', '.npy')
            im = cv2.imread(files[i], -1)
            np.save(save_folder1 + '/' + name, im)
            np.save(save_folder2 + '/' + name, im)
            i += 1

        else:
            
            name = files[i].split('/')[-1].replace('.tif', '.npy')
            images = []
            while files[i][-7:-4] == 'C04':
                im = cv2.imread(files[i],-1)
                #images.append(files[i].split('/')[-1])
                images.append(im)
                i += 1

                if i == len(files):
                    break

            images = np.array(images).transpose(1,2,0)
            np.save(save_folder1 + '/' + name, images)
            np.save(save_folder2 + '/' + name, images)

        

