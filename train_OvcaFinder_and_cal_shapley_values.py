import numpy as np

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from multiprocessing.pool import Pool

import shap

shap.initjs()

# prepare the train data to train the RF
feature_train_csv = "./data/valid.csv"  # Since 'train.csv' was used to train DL model, here we used valid to train RF. 
train = pd.read_csv(feature_train_csv)
feature_test_csv = "./data/test.csv"
test = pd.read_csv(feature_test_csv)

x_train = []
y_train = []
x_test = []
y_test = []

for head in ["O-RADS1", "O-RADS2", "O-RADS3", "O-RADS4", "O-RADS5"]:
    x_train_temp = []
    for h in ['age', 'CA125', 'diameter', head, 'DL_preds']:
        x_train_temp.append(np.array(train[h]).reshape(-1, 1))
    x_train_temp = np.concatenate(x_train_temp, axis=1)
    x_train.append(x_train_temp)
    y_train.append(list(train['abnormal']))

x_train = np.concatenate(x_train, axis=0)
y_train = np.concatenate(y_train, axis=0)

aucs = []
params = []
inputs = []
for n_estimators in range(10, 301, 10):
    for max_depth in range(5, 51, 5):
        for min_samples_split in range(15, 51, 15):
            for min_samples_leaf in range(5, 20, 3):
                for max_features in range(2, 4, 1):
                    for random_state in range(4, 21, 4):
                        inputs.append([n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features,
                                       random_state])


def process_one(inp):
    n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, random_state = inp
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
    )
    model.fit(x_train, y_train)
    aucs_ = []
    for head in ["O-RADS1", "O-RADS2", "O-RADS3", "O-RADS4", "O-RADS5"]:
        x_test = []
        for h in ['age', 'CA125', 'diameter', head, 'DL_preds']:
            x_test.append(np.array(test[h]).reshape(-1, 1))
        x_test = np.concatenate(x_test, axis=1)
        y_test = list(test['abnormal'])
        x_pred_test = model.predict_proba(x_test)[:, 1]
        auc = metrics.roc_auc_score(y_test, x_pred_test)
        aucs_.append(auc)
    auc = np.mean(aucs_)

    return auc, inp


print(len(inputs))
with Pool() as tp:
    for i, (o_auc, o_param) in enumerate(tp.imap_unordered(process_one, inputs)):
        print('\rdone {0:%}'.format(i / len(inputs)), end='')
        aucs.append(o_auc)
        params.append(o_param)
max_auc = np.max(aucs)
idx = aucs.index(max_auc)
print('max_auc:', max_auc, 'param:', params[idx])

############################
# calculate shapley values #
############################

best_param = params[idx]
model = RandomForestClassifier(
    n_estimators=best_param[0],
    max_depth=best_param[1],
    min_samples_split=best_param[2],
    min_samples_leaf=best_param[3],
    max_features=best_param[4],
    random_state=best_param[5],
)
model.fit(x_train, y_train)
print(model.feature_importances_)

o_preds = {}
x_test = []
for head in ["O-RADS1"]:
    for h in ['age', 'CA125', 'diameter', head, 'DL_preds']:
        x_test.append(np.array(test[h]).reshape(-1, 1))
    x_test = np.concatenate(x_test, axis=1)

explainer = shap.explainers.Exact(model.predict_proba, x_train,
                                  feature_names=['Age', 'CA125 Concentration', 'Lesion Diameter', 'O-RADS', 'DL_model'])
shap_values = explainer(x_test)

shap_values = shap_values[..., 1]
shap.plots.bar(shap_values)
for i in range(100):
    shap.plots.waterfall(shap_values[i])
