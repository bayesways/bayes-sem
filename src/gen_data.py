import numpy as np
import pandas as pd
import pystan
import datetime
import sys
import os
from scipy.stats import norm, multivariate_normal
from codebase.data import gen_data

from codebase.file_utils import save_obj, load_obj
import argparse

parser = argparse.ArgumentParser()
# Optional arguments
parser.add_argument("-rs", "--random_seed", help="random seed to use for data generation", type=int, default=None)
parser.add_argument("-th", "--task_handle", help="handle (name) for task", type=str, default="_")

args = parser.parse_args()

print("\n\nCreating directory")
nowstr = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_') # ISO 8601 format
log_dir =  "./log/"+nowstr+"%s/" % args.task_handle

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

print("\n\nGenerating data")
data = gen_data(500, 6, 2, random_seed = args.random_seed)
print("\n\nN = %d, J= %d, K =%d"%(data['N'],data['J'], data['K'] ))
print("\n\n********** \nReplacing last 10% of data with 10's \n***********\n\n")
data['y'][-50:] = 3
print("\nSaving data\n\n")
save_obj(data, 'data', log_dir)
