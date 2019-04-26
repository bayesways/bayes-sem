
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


df = pd.read_csv("../dat/muthen_women.csv")

# In[3]:


data = dict()
data['N'] = df.shape[0]
data['K'] = 5
data['J'] = df.shape[1]
data['y'] = df.values


# In[4]:


stan_data = dict(N = data['N'], K = data['K'], J = data['J'], yy = data['y'])


# In[5]:


with open('./codebase/stan_code/cont/CFA/marg_m.stan', 'r') as file:
    model_code = file.read()
print(model_code)


# In[6]:


sm = pystan.StanModel(model_code=model_code, verbose=False)


# In[7]:


nowstr = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_') # ISO 8601 format
task_id = 'CFA_NN_marg_muthen_women'
log_dir =  "./log/"+nowstr+"%s/" % task_id


# In[8]:


if not os.path.exists(log_dir):
    os.makedirs(log_dir)


# In[10]:


num_chains = 1
num_samples = 1000
num_warmup = 1000
num_iter = num_samples + num_warmup


# In[11]:


fit_run = sm.sampling(data=stan_data, iter=num_iter, warmup=num_warmup, chains=num_chains)


# In[12]:


save_obj(sm, 'sm', log_dir)
save_obj(fit_run, 'fit', log_dir)
fit=fit_run
