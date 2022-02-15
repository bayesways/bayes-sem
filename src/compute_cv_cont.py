import numpy as np
from codebase.model_fit_cont import get_energy_scores, get_log_score
from codebase.post_process import remove_cn_dimension
from codebase.file_utils import save_obj, load_obj
import argparse
from pdb import set_trace

parser = argparse.ArgumentParser()
parser.add_argument("logdir", help="path to files", type=str, default=None)

# Optional arguments
parser.add_argument("-nsim", "--nsim_ppp",
                    help="number of posterior samples to use for PPP", type=int, default=4000)

args = parser.parse_args()

logdir = args.logdir
if logdir[-1] != "/":
    logdir = logdir + "/"
############################################################
################ Load Model Data  ##########
complete_data = load_obj("complete_data", logdir)
ps = dict()
ps[0] = load_obj('ps_0', logdir)
ps[1] = load_obj('ps_1', logdir)
ps[2] = load_obj('ps_2', logdir)
mcmc_length = ps[0]['alpha'].shape[0]
num_chains = ps[0]['alpha'].shape[1]

for fi in range(3):
    for name in ps[fi].keys():
        ps[fi][name] = remove_cn_dimension(ps[fi][name])

Ds = dict()
for fold_index in range(3):
    # Ds[fold_index] = get_energy_scores(
    #     ps[fold_index],
    #     complete_data[fold_index]['test']['yy'],
    #     args.nsim_ppp,
    #     )
    Ds[fold_index] = get_log_score(
        ps[fold_index],
        complete_data[fold_index]['test']['yy'],
        args.nsim_ppp,
        )

# ###########################################################
# ############### Compare CV scores  ##########
# print('\nFold Sum %.2f'%np.sum(Ds))
# for f in range(3):
#     print('Fold %.2f'%Ds[f])


###########################################################
############### Compare CV scores  ##########
# score_names = ['variogram']
score_names = ['logscore']
for name in score_names:
    a = [Ds[fold][name] for fold in range(3)]
    print("\n%s Fold Sum %.2f" % (name, np.sum(a)))

