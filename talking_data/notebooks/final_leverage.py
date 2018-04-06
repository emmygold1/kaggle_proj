
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd


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

train_usecols = ['ip', 'app', 'device', 'os', 'channel', 'is_attributed']


# In[4]:

train = pd.read_csv('../data/train.csv', dtype=dtypes, usecols=train_usecols)


# In[5]:

train.head()


# In[6]:

train_sample = pd.read_csv('../data/train_sample.csv', dtype=dtypes, usecols=train_usecols)


# In[7]:

test = pd.read_csv('../data/test.csv', dtype=dtypes, usecols=train_usecols[:-1])


# In[8]:

test_sample = pd.read_csv('../data/test_supplement.csv', dtype=dtypes, usecols=train_usecols[:-1])


# In[9]:

test = pd.concat([test, test_sample])


import pickle


# In[13]:

with open('../results/random_forest_no_feat_eng_{}.pk'.format(10), 'rb') as f:
    rfc = pickle.load(f)


pred = rfc[2].predict(test)
test.loc[:, 'is_attributed'] = pred
train = pd.concat([train, test])
del pred
del test
del test_sample
del train_sample
import gc
gc.collect()
rfc = rfc[2]
rfc.fit(train.iloc[:, :-1], train.iloc[:, -1])
with open('../results/final_leverage.pk', 'wb') as f:
    pickle.dump(rfc, f)
