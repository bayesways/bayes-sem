import numpy as np
from codebase.file_utils import load_obj
from codebase.model_fit_bin import get_PPP
from codebase.post_process import remove_cn_dimension

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("logdir", help="path to files", type=str, default=None)

# Optional arguments
parser.add_argument(
    "-nsim",
    "--nsim_ppp",
    help="number of posterior samples to use for PPP",
    type=int,
    default=50,
)

args = parser.parse_args()

log_dir = args.logdir
if log_dir[-1] != "/":
    log_dir = log_dir + "/"


data = load_obj("data", log_dir)
ps = load_obj("ps", log_dir)
num_chains = ps["alpha"].shape[1]
num_samples = ps["alpha"].shape[0]

for name in ["yy", "beta", "alpha"]:
    ps[name] = remove_cn_dimension(ps[name])
PPP_vals = get_PPP(data, ps, args.nsim_ppp)

ppp = np.sum(PPP_vals[:, 0] < PPP_vals[:, 1]) / args.nsim_ppp
print("PPP %.2f" % ppp)
