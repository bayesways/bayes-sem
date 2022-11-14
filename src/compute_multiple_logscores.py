from codebase.file_utils import save_obj, load_obj
import numpy as np
from codebase.model_fit_cont import get_log_score
from codebase.post_process import remove_cn_dimension
import datetime
import os
from tqdm import tqdm


def get_cv_score_for_i(log_dir, i, nsim_ppp=1000):
    complete_data = load_obj("complete_data" + str(i), log_dir)
    ps = dict()
    ps[0] = load_obj("ps" + str(i) + "_0", log_dir)
    ps[1] = load_obj("ps" + str(i) + "_1", log_dir)
    ps[2] = load_obj("ps" + str(i) + "_2", log_dir)

    for fi in range(3):
        for name in ps[fi].keys():
            ps[fi][name] = remove_cn_dimension(ps[fi][name])

    Ds = dict()
    for fold_index in range(3):
        Ds[fold_index] = get_log_score(
            ps[fold_index],
            complete_data[fold_index]["test"]["yy"],
            nsim_ppp,
        )

    ###########################################################
    ############### Compare CV scores  ##########
    score_names = ["logscore"]
    for name in score_names:
        a = [Ds[fold][name] for fold in range(3)]
        cvscore = np.sum(a)
    #     return cvscore, np.round(a, 3)
    return cvscore


def get_all_logscores(log_dir, nsim=100):
    total_scores = np.empty(nsim)
    #     fold_scores =np.empty((nsim,3))
    for i in tqdm(range(nsim)):
        total_scores[i] = get_cv_score_for_i(log_dir, i)
    return total_scores


############################################################
###### Create Directory or Open existing ##########
nowstr = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_")  # ISO 8601 format
log_dir = "./log/revision_runs/" + nowstr
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
if log_dir[-1] != "/":
    print("\n\nAppending `/`-character at the end of directory")
    log_dir = log_dir + "/"

print("\n\nSaving results in %s" % log_dir)


############################################################
# Comput Logscores ##########

model_logscores = dict()

log_dir1 = "./log/revision_runs/20221112_213907_mult_m2_s3_cv/"
model_logscores["AZ"] = get_all_logscores(log_dir1)
save_obj(model_logscores, "model_logscores", log_dir)

log_dir2 = "./log/20221112_213918_mult_m4_s3_cv/"
model_logscores["EFA"] = get_all_logscores(log_dir2)
save_obj(model_logscores, "model_logscores", log_dir)

log_dir3 = "./log/20221112_214124_mult_m5_s3_cv/"
model_logscores["EFA"] = get_all_logscores(log_dir3)
save_obj(model_logscores, "model_logscores", log_dir)
