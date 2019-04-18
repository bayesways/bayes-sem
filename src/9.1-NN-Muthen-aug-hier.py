
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import pystan
from scipy.stats import norm, multivariate_normal, invwishart, invgamma
from statsmodels.tsa.stattools import acf
import datetime
import sys
import os

from codebase.plot import *
from codebase.file_utils import save_obj, load_obj
from codebase.post_process import *


df = pd.read_csv("../dat/muthen_men.csv")

data = dict()
data['N'] = df.shape[0]
data['K'] = 5
data['J'] = df.shape[1]
data['y'] = df.values


stan_data = dict(N = data['N'], K = data['K'], J = data['J'], yy = data['y'])


with open('./codebase/stan_code/cont/CFA/aug_hier_muthen.stan', 'r') as file:
    model_code = file.read()
print(model_code)


sm = pystan.StanModel(model_code=model_code, verbose=False)


nowstr = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_') # ISO 8601 format
task_id = 'CFA_NN_aug_hier_muthen_men'
log_dir =  "./log/"+nowstr+"%s/" % task_id


# In[9]:


if not os.path.exists(log_dir):
    os.makedirs(log_dir)


# In[11]:


num_chains = 1
num_samples = 1000
num_warmup = 1000
num_iter = num_samples + num_warmup


fit_run = sm.sampling(data=stan_data, iter=num_iter, chains=num_chains)

save_obj(sm, 'sm', log_dir)
save_obj(fit_run, 'fit', log_dir)
