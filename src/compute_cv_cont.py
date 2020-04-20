import numpy as np
import pandas as pd
import pystan
import datetime
import sys
import os
from scipy.stats import multivariate_normal
from sklearn.model_selection import KFold
from codebase.model_fit_cont import Nlogpdf


from codebase.file_utils import save_obj, load_obj
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("logdir", help="path to files", type=str, default=None)

# Optional arguments
parser.add_argument("-prm", "--print_model",
                    help="print model on screen", type=int, default=1)

args = parser.parse_args()

print("\n\nPrinting Stan model code \n\n")


logdir = args.logdir
if logdir[-1] != "/":
    print("\n\nAppending `/`-character at the end of directory")
    logdir = logdir + "/"

############################################################
################ Load Data and models  ##########

print("\n\nLoading files...\n\n")

complete_data = load_obj("complete_data", logdir)


model_posterior_samples = dict()

model_posterior_samples = dict()
model_posterior_samples[0] = load_obj('ps_0', logdir)
model_posterior_samples[1] = load_obj('ps_1', logdir)
model_posterior_samples[2] = load_obj('ps_2', logdir)

print("\n\nComputing Folds...\n\n")

mcmc_length = model_posterior_samples[0]['alpha'].shape[0]
cn = 0

Ds = np.empty((mcmc_length, 3))
for fold_index in range(3):
    for mcmc_iter in range(mcmc_length):
        model_lgpdf = Nlogpdf(complete_data[fold_index]['test']['yy'],
                              model_posterior_samples[fold_index]['alpha'][mcmc_iter, cn],
                              model_posterior_samples[fold_index]['Marg_cov'][mcmc_iter, cn])

        Ds[mcmc_iter, fold_index] = -2*np.sum(model_lgpdf)

print(Ds.shape)
save_obj(Ds, 'ds', "./")
print(np.sum(Ds, axis=0)/mcmc_length)
result = np.round(100*np.mean(np.sum(Ds, axis=0)/mcmc_length))
print("k-fold Index = %d %%" % result)
