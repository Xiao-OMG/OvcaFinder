import cv2
import os
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from multiprocessing.pool import Pool


def process_one(img_path):
    img = cv2.imread(img_path)
    cam = np.zeros([img.shape[0], img.shape[1]])
    for j, camdir in enumerate(camdirs):
        cam_path = os.path.join(camdir, s, os.path.basename(img_path)[:-4] + '.npy')
        cam_temp = np.load(cam_path)
        cam_temp_temp = np.zeros(cam_temp.shape)
        for m in range(cam_temp.shape[0]):
            for n in range(cam_temp.shape[1]):
                cam_temp_temp[m, n] = cam_temp[m, n]

        cam_temp = cam_temp_temp
        img_shape = (img.shape[1], img.shape[0])
        cam_temp = cv2.resize(cam_temp, img_shape)
        cam = cam_temp + cam
    cam = cam / float(len(camdir))
    cam[cam <= thresh] = thresh
    fig, axs = plt.subplots(figsize=(img.shape[1] / 100, img.shape[0] / 100))
    plt.axis('off')
    plt.imshow(cam, cmap='jet', alpha=0.7)
    plt.imshow(img, alpha=0.3)

    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    save_name = os.path.join(cam_jpg_dir, os.path.basename(img_path))
    os.makedirs(os.path.dirname(save_name), exist_ok=True)
    plt.savefig(save_name, pad_inches=0)
    plt.close(fig)


camdirs = [
    'cams/cam_mask_densenet121',
    'cams/cam_mask_densenet169',
    'cams/cam_mask_densenet201',
    'cams/cam_mask_resnet34',
    'cams/cam_mask_efficientnet_b5',
    'cams/cam_mask_efficientnet_b6',
]

cam_jpg_dir = 'cams/merge_cam_mask'

thresh = 0.5
csvfile = "./data/valid.csv"
paths = list(pd.read_csv(csvfile)['path'])

num_tasks = len(paths)
print(num_tasks)
with Pool() as tp:
    for i, _ in enumerate(tp.imap_unordered(process_one, paths)):
        print('\rdone {0:%}'.format(i / num_tasks), end='')
