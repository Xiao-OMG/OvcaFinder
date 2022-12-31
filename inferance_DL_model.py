from fastai.vision import *
import os
import pandas as pd

from train_DL_model import mean_absolute_error_, SaveCallback


def infer_csv_file(model_name, csv_add, sv_add=None):
    img_add = pd.read_csv(csv_add)
    learn = load_learner(model_path, model_name, test=ImageList.from_csv('/', csv_add), bs=16)
    result = learn.get_preds(ds_type=DatasetType.Test)
    preds, y = result[0], result[1]
    np_preds = preds.sigmoid().numpy()
    img_add['DL_preds'] = np_preds
    img_add['path'] = [h for h in img_add['path']]
    pred_sv_dir = os.path.join(test_result_dir, model_name.split('.')[0])
    if not os.path.exists(pred_sv_dir):
        os.makedirs(pred_sv_dir)
    if not sv_add:
        sv_add = os.path.join(pred_sv_dir, csv_add.split('/')[-1])
    img_add.to_csv(sv_add, index=False)


test_result_dir = './results'

model_dict = [
    ['ckpt/densenet121/', '{model file name}.pkl'],
    ['ckpt/densenet169', '{model file file name}.pkl'],
    ['ckpt/densenet201/', '{model name}.pkl'],
    ['ckpt/resnet34/', '{model file name}.pkl'],
    ['ckpt/efficientnet_b5/', '{model file name}.pkl'],
    ['ckpt/efficientnet_b6/', '{model file name}.pkl'],
]

for i in range(len(model_dict)):
    model_path = model_dict[i][0]
    model_name = model_dict[i][1]
    infer_csv_file(model_name, "./data/test.csv")
