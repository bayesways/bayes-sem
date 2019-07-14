import numpy as np
import pandas as pd
import pystan
import datetime
import sys
import os
from scipy.stats import multivariate_normal
from sklearn.model_selection import KFold

from codebase.file_utils import save_obj, load_obj
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("logdir_1", help="path to files", type=str, default=None)
parser.add_argument("logdir_2", help="path to files", type=str, default=None)

# Optional arguments
parser.add_argument("-prm", "--print_model", help="print model on screen", type=int, default=1)

args = parser.parse_args()

print("\n\nPrinting Stan model code \n\n")


log_dir1 = args.logdir_1
if log_dir1[-1] != "/":
    print("\n\nAppending `/`-character at the end of directory")
    log_dir1 = log_dir1+ "/"

if bool(args.print_model):
    file = open(log_dir1+"model.txt", "r")
    print(file.read())
    file.close()

log_dir2 = args.logdir_2
if log_dir2[-1] != "/":
    print("\n\nAppending `/`-character at the end of directory")
    log_dir2 = log_dir2+ "/"

if bool(args.print_model):
    print("\n\nModel 1...")
    file = open(log_dir1+"model.txt", "r")
    print(file.read())
    file.close()

    print("\n\nModel 2...")
    file = open(log_dir2+"model.txt", "r")
    print(file.read())
    file.close()


############################################################
################ Load Data and models  ##########

print("\n\nLoading files...\n\n")


print("\n\nChecking data integrity...\n\n")
complete_data = load_obj("complete_data", log_dir1)
complete_data2 = load_obj("complete_data", log_dir2)

for fold_index in range(3):
    np.testing.assert_equal(complete_data[fold_index]['test']['yy'],\
        complete_data2[fold_index]['test']['yy'] )
    np.testing.assert_equal(complete_data[fold_index]['train']['yy'],\
        complete_data2[fold_index]['train']['yy'] )



model_posterior_samples = dict()

ps = dict()
ps[0] = load_obj('ps_0', log_dir1)
ps[1] = load_obj('ps_1', log_dir1)
ps[2] = load_obj('ps_2', log_dir1)
model_posterior_samples[1] = ps

ps = dict()
ps[0] = load_obj('ps_0', log_dir2)
ps[1] = load_obj('ps_1', log_dir2)
ps[2] = load_obj('ps_2', log_dir2)
model_posterior_samples[2] = ps


def Nlogpdf(yy, mean, cov):
    return multivariate_normal.logpdf(yy,
                               mean,
                               cov)
sample_mean = np.mean(complete_data[0]['train']['yy'], axis=0)
sample_cov = np.cov(complete_data[0]['train']['yy'], rowvar=False)


print("\n\nComputing Folds...\n\n")

mcmc_length = ps[0]['alpha'].shape[0]
Ds = np.empty((mcmc_length,))
for mcmc_iter in range(mcmc_length):
    model_1_lgpdf = Nlogpdf(complete_data[0]['test']['yy'],
        model_posterior_samples[1][0]['alpha'][mcmc_iter],
        model_posterior_samples[1][0]['Marg_cov'][mcmc_iter])

    model_2_lgpdf = Nlogpdf(complete_data[0]['test']['yy'],
        model_posterior_samples[2][0]['alpha'][mcmc_iter],
        model_posterior_samples[2][0]['Marg_cov'][mcmc_iter])
    Ds[mcmc_iter] = -2*np.sum(model_1_lgpdf - model_2_lgpdf)

print(Ds)
print(Ds.shape)
print(np.mean(Ds, axis=0))
