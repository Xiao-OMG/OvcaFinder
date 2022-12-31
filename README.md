# OvcaFinder code

## Installization
python3.6 and fastai
```
pip install -r requirements.txt
```

## Data preparement
change your custom data into csv format and put them into *./data* folder

## Train DL model
select which backbone u want to train at line 85: 

'densenet121' / 'densenet169' / 'densenet201' / 'efficientnet_b5' / 'efficientnet_b6' / 'resnet34'
```
85  model_name = 'densenet121'
```
then
```
python train_DL_model.py
```

## Inferance DL model
```
python inferance_DL_model.py
```

## Generate Grad-CAM mask
```
python generate_grad_CAM.py
python draw_merged_CAM.py
```

## Train OvcaFinder
```
python train_OvcaFinder_and_cal_shapley_values.py
```

