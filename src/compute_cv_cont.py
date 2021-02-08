import numpy as np
from codebase.model_fit_cont import get_energy_scores
from codebase.file_utils import save_obj, load_obj
import argparse
from pdb import set_trace

parser = argparse.ArgumentParser()
parser.add_argument("logdir", help="path to files", type=str, default=None)

# Optional arguments
parser.add_argument("-nsim", "--nsim_ppp",
                    help="number of posterior samples to use for PPP", type=int, default=1000)

args = parser.parse_args()

logdir = args.logdir
if logdir[-1] != "/":
    logdir = logdir + "/"
############################################################
################ Load Model Data  ##########
complete_data = load_obj("complete_data", logdir)
ps = dict()
ps = dict()
ps[0] = load_obj('ps_0', logdir)
ps[1] = load_obj('ps_1', logdir)
ps[2] = load_obj('ps_2', logdir)
mcmc_length = ps[0]['alpha'].shape[0]
num_chains = ps[0]['alpha'].shape[1]
Ds = np.empty((3, num_chains))
for fold_index in range(3):
    for cn in range(num_chains):
        Ds[fold_index, cn] = get_energy_scores(
            ps[fold_index],
            complete_data[fold_index]['test']['yy'],
            args.nsim_ppp,
            cn
            )

###########################################################
############### Compare CV scores  ##########
fold_chain_average_matrix = np.mean(Ds,axis=1)
print('\nChain/Fold Average %.2f'%np.mean(fold_chain_average_matrix))
for f in range(3):
    chain_scores = fold_chain_average_matrix[f]
    print("\nFold %d Avg =  %.2f"%(
        f,
        np.mean(chain_scores))
        )
    print(chain_scores)
