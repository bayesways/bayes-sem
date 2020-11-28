import numpy as np
from codebase.model_fit_cont import get_lgscr
from codebase.file_utils import save_obj, load_obj
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("logdir_model", help="path to files", type=str, default=None)
parser.add_argument("logdir_benchmark", help="path to files", type=str, default=None)

# Optional arguments
parser.add_argument("-nsim", "--nsim_ppp",
                    help="number of posterior samples to use for PPP", type=int, default=100)


args = parser.parse_args()

print("\n\nPrinting Stan model code \n\n")


logdir_model = args.logdir_model
if logdir_model[-1] != "/":
    print("\n\nAppending `/`-character at the end of directory")
    logdir_model = logdir_model + "/"

############################################################
################ Load Model Data  ##########

print("\n\nLoading files...\n\n")

complete_data = load_obj("complete_data", logdir_model)


model_ps = dict()

model_ps = dict()
model_ps[0] = load_obj('ps_0', logdir_model)
model_ps[1] = load_obj('ps_1', logdir_model)
model_ps[2] = load_obj('ps_2', logdir_model)

print("\n\nComputing FolDs_model...\n\n")

mcmc_length = model_ps[0]['alpha'].shape[0]
num_chains = model_ps[0]['alpha'].shape[1]


Ds_model = np.empty((3, args.nsim_ppp, num_chains))
for fold_index in range(3):
    Ds_model[fold_index] = get_lgscr(
        model_ps[fold_index],
        complete_data[fold_index],
        args.nsim_ppp
        )


logdir_benchmark = args.logdir_benchmark
if logdir_benchmark[-1] != "/":
    print("\n\nAppending `/`-character at the end of directory")
    logdir_benchmark = logdir_benchmark + "/"


############################################################
################ Load Benchmark Data  ##########


print("\n\nLoading files...\n\n")

complete_data = load_obj("complete_data", logdir_benchmark)
benchmark_ps = dict()
benchmark_ps = dict()
benchmark_ps[0] = load_obj('ps_0', logdir_benchmark)
benchmark_ps[1] = load_obj('ps_1', logdir_benchmark)
benchmark_ps[2] = load_obj('ps_2', logdir_benchmark)

print("\n\nComputing FolDs_benchmark...\n\n")

mcmc_length = benchmark_ps[0]['alpha'].shape[0]
num_chains = benchmark_ps[0]['alpha'].shape[1]

Ds_benchmark = np.empty((3, args.nsim_ppp, num_chains))
for fold_index in range(3):
    Ds_benchmark[fold_index] = get_lgscr(
        benchmark_ps[fold_index],
        complete_data[fold_index],
        args.nsim_ppp
        )

fold_chain_average_matrix = np.mean(Ds_model>Ds_benchmark, 1)
print('Chain/Fold Average %.2f'%np.mean(fold_chain_average_matrix))
for f in range(3):
    chain_scores = fold_chain_average_matrix[f]
    print("\nFold %d Avg =  %.2f"%(
        f,
        np.mean(chain_scores))
        )
    print(chain_scores)
