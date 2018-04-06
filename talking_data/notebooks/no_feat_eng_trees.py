
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


# In[2]:

dtypes = {
    'ip'            : 'uint32',
    'app'           : 'uint16',
    'device'        : 'uint16',
    'os'            : 'uint16',
    'channel'       : 'uint16',
    'is_attributed' : 'uint8',
    'click_id'      : 'uint32'
}


# In[3]:

train_usecols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']


# In[4]:

train = pd.read_csv('../data/train.csv', dtype=dtypes, usecols=train_usecols)


train.loc[:, 'click_time'] = pd.to_datetime(train.click_time, format='%Y-%m-%d %H:%M:%S')



trees = [10, 50, 100, 150]
split_time = '2017-11-09 00:00:00'


for tree in trees:
    print(tree)
    rfc = RandomForestClassifier(n_estimators=25, class_weight={0: 1, 1: 100}, n_jobs=4)
    rfc.fit(train.loc[train.click_time < split_time].iloc[:, :-2], train.loc[train.click_time < split_time].iloc[:, -1])
    pred_prob = rfc.predict_proba(train.loc[train.click_time >= split_time].iloc[:, :-2])[:, 1]
    with open('../results/random_forest_no_feat_eng_tree_{}.pk'.format(tree), 'wb') as f:
        pickle.dump((tree, roc_auc_score(train.loc[train.click_time >= split_time].iloc[:, -1], pred_prob), rfc), f)
