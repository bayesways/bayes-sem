import numpy as np
import pandas as pd
import pystan
import datetime
import sys
import os
import numpy as np
from sklearn.model_selection import KFold

from codebase.file_utils import save_obj, load_obj
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("logdir", help="path to files", type=str, default=None)
# Optional arguments
parser.add_argument("-prm", "--print_model", help="print model on screen", type=int, default=1)

args = parser.parse_args()

print("\n\nPrinting Stan model code \n\n")


log_dir = args.logdir
if log_dir[-1] != "/":
    print("\n\nAppending `/`-character at the end of directory")
    log_dir = log_dir+ "/"

if bool(args.print_model):
    file = open(log_dir+"model.txt", "r")
    print(file.read())
    file.close()

############################################################
################ Load Data and models  ##########

print("\n\nLoading files...\n\n")
complete_data = load_obj("complete_data", log_dir)

ps = dict()
ps[0] = load_obj('ps_0', log_dir)
ps[1] = load_obj('ps_1', log_dir)
ps[2] = load_obj('ps_2', log_dir)


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
    Ds[mcmc_iter] = -2*np.sum(\
        Nlogpdf(complete_data[0]['test']['yy'], ps_model3_0['alpha'][mcmc_iter], ps_model3_0['Sigma'][mcmc_iter] ) -\
        Nlogpdf(complete_data[0]['test']['yy'], sample_mean, sample_cov ))

print(Ds)
print(Ds.shape)
print(np.mean(Ds, axis=0))
