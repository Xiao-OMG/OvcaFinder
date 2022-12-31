from torchvision.models import densenet169, densenet121, densenet201, resnet34
from timm import create_model
import torch
import pandas as pd
import os
from fastai.vision import *

from sklearn.metrics import roc_auc_score

trainfile_add = "./data/train.csv"
testfile_add = "./data/valid.csv"

full_dfs = []

full_train_df = pd.read_csv(trainfile_add)
full_train_df['train_valid'] = False
full_dfs.append(full_train_df)

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


def validation_eval(learn):
    valid_preds = learn.get_preds(ds_type=DatasetType.Valid)
    acts = full_test_df['abnormal'].values
    preds = valid_preds[0].sigmoid().view(-1).numpy()
    try:
        auc_score = roc_auc_score(acts, preds)
    except Exception as e:
        print(e)
        auc_score = 0.5
    print(f'2cls auc: {auc_score:.4}')
    return auc_score


class SaveCallback(LearnerCallback):
    _order = 99

    def __init__(self, learn):
        super().__init__(learn)
        self.epoch = 0
        self.skip = False

    def on_epoch_end(self, **kwargs):
        self.epoch += 1
        if self.skip: return
        avg_auc = validation_eval(self.learn)
        learn.save('%s-epoch-%d-AUC-%.4f' % (model_name, self.epoch, avg_auc))
        learn.export('%s-epoch-%d-AUC-%.4f.pkl' % (model_name, self.epoch, avg_auc))


def mean_absolute_error_(pred: Tensor, targ: Tensor) -> Rank0Tensor:
    "Mean absolute error between `pred` and `targ`."
    pred, targ = flatten_check(pred, targ)

    return torch.abs(targ - pred.sigmoid()).mean()


def timm_model_effb5(pretrained):
    return create_model('tf_efficientnet_b5', pretrained=pretrained,
                        num_classes=0, global_pool


def timm_model_effb6(pretrained):
    return create_model('tf_efficientnet_b6', pretrained=pretrained,
                        num_classes=0, global_pool='')


img_size = 512
model_name = 'densenet121'  # 'densenet121' / 'densenet169' / 'densenet201' / 'efficientnet_b5' / 'efficientnet_b6' / 'resnet34'
model_ckpt = './ckpt/' + model_name  # 
os.makedirs(model_ckpt, exist_ok=True)

if model_name == 'densenet121':
    model = densenet121
elif model_name == 'densenet169':
    model = densenet169
elif model_name == 'densenet201':
    model = densenet201
elif model_name == 'efficientnet_b5':
    model = timm_model_effb5
elif model_name == 'efficientnet_b6':
    model = timm_model_effb6
elif model_name == 'resnet34':
    model = resnet34

data = get_data(img_size, get_src(full_df), bs=16)
data.c = 1

learn = cnn_learner(data, densenet121, pretrained=True, metrics=[mean_absolute_error_],
                    path=model_path,
                    callback_fns=[SaveCallback],
                    loss_func=BCEWithLogitsFlat(),
                    )

learn.model = torch.nn.DataParallel(learn.model)
lr = 1e-3
learn.unfreeze()
learn.lr_find()
learn.fit_one_cycle(40, max_lr=slice(lr))
