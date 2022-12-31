import os
import torch
import numpy as np
import cv2
from matplotlib import pyplot as plt
import pandas as pd

from timm import create_model
from fastai.vision import *
from fastai.callbacks.hooks import *

testfile_add = "./data/valid.csv"
full_dfs = []
full_test_df = pd.read_csv(testfile_add)
full_test_df['train_valid'] = True
full_dfs.append(full_test_df)
full_df = pd.concat(full_dfs)


def get_src(df=full_df):
    return (ImageList
            .from_df(df, '/')
            .split_from_df('train_valid')
            .label_from_df('abnormal'))


def get_data(size, src, bs=16):
    trfms = get_transforms(do_flip=True, flip_vert=False, max_rotate=30, max_zoom=1.2,
                           max_lighting=0.25, max_warp=0.15, p_affine=0.15, p_lighting=0.15, )
    return src.transform(trfms, size=size).databunch(bs=bs, num_workers=56)


img_size = 512
data = get_data(img_size, get_src(full_df), bs=16)
data.c = 1
model = torch.load('ckpt/densenet121/{model file name}.pkl')['model'].module
model = model.cuda()
model.eval()

save_cam_mask_dir = 'cams/cam_mask_densenet121/'
os.makedirs(save_cam_mask_dir, exist_ok=True)

for nn in range(len(data.valid_ds)):
    print('%d|%d' % (nn, len(data.valid_ds)))
    x, y = data.valid_ds[nn]
    x_path = data.valid_ds.x.inner_df.path[nn]

    img = cv2.imread(x_path)
    # img = cv2.resize(img, (load_size, load_size))
    # m = learn.model.eval()
    xb, _ = data.one_item(x)
    xb = xb.cuda()
    with hook_output(model[0]) as hook_a:
        with hook_output(model[0], grad=True) as hook_g:
            preds = model(xb)
            preds[0, 0].backward()
    acts = hook_a.stored[0].cpu()
    ax = plt.subplot()
    cam = acts.mean(0).numpy()
    cam *= torch.sigmoid(preds[0, 0]).cpu().item()
    np.save(os.path.join(save_cam_mask_dir, '%s.npy' % os.path.basename(x_path)[:-4]), cam)
