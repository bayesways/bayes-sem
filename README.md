# Factor Analysis

## Setup

Initial install:

    $ conda env create -f environment.yml
    $ source activate bayes-sem

Update:

    $ conda env update -f environment.yml

Deactivate Environment:

    $ source deactivate

Remove:

    $ conda remove -n bayes-sem --all


Convert `a.ipynb` to `a.py`:

    $ jupyter nbconvert --to script a.ipynb


## How to use:

* To run the Muthen model run `run_muthen_exp1.py`. The command runs as follows

  ```
  run_muthen_exp1.py <num_warmup> <num_samples> <num_chains> <men/women> <model_code>
  ```

  with optional flags for `-th` (`--task_handle`), `-pm` (`--print_model`), and
  `-xdir` (`--existing_directory`). The results are saved in
  `src/log/<date-time>_<task_handle>`.
  
  To choose which model to run use the `<model_code>` option as follows:
  "0:full model, 1:no u's, 2: no u's no approx zero betas "
  
  If an existing directory is given then the script looks for an existing compiled
  stan model to load and run with the new number of iterations.
  
  The results are processed using `model-results.py`

* The same model but replacing 10% of the data with 10's `run_muthen_exp2.py`.
The command runs as follows

    ```
    run_muthen_exp2.py <num_warmup> <num_samples> <num_chains> <men/women>
    ```

* Script `muthen_exp_ppp_kfold.py` runs a kfold version of the `run_muthen_exp1`
script. It works as follows

    ```
    run_muthen_exp1.py <num_warmup> <num_samples> <num_chains> <men/women> <model_code>
    ```

  with optional flags for `-th` (`--task_handle`), `-pm` (`--print_model`), `-nfl`
  (`n_splits`) and `-xdir` (`--existing_directory`). The results are saved in
  `src/log/<date-time>_<task_handle>`.
  
  To choose which model to run use the `<model_code>` option as follows:
  "0:full model, 1:no u's, 2: no u's no approx zero betas "
  
  If an existing directory is given then the script looks for an existing compiled
  stan model to load and run with the new number of iterations.
  
  The results are processed using `kfold_results.py`


### On Fabian

We installed pystan version 2.19 which should run the same scripts as local correctly.

1. Activate `apps/anaconda3` and use env `pystan-dev`

    ```
    module add apps/anaconda3
    source activate pystan-dev
    ```
    
2. Use `python3` at front and use the fabian version of any script (ends in `_fabian.py`). For example

    ```
    python3 run_muthen_exp1_fabian.py <num_warmup> <num_samples> <num_chains> <men/women> <model_code>
    ```

## Results:

* To run the experiment of Muthen data `run_muthen_exp1.py`. Usually we run this
on the server, collect the results and visualize them locally with
`1.1.Muthen-women` or `1.2.Muthen-men`. Also see more recent results for
`women` only run on fabian at
`10.Muthen-results-model0`, `10.Muthen-results-model1`, and `10.Muthen-results-model2`
respectively for different models. 

* We ran the following models to compare the PPP values using `model-results.py`.
Results are saved in `log/fabian_runs` folders:
 
    * `model0` : Factor model with u's and approx zeros
    * `model2` : Factor model with no u's and exact zeros

* In addition we ran the kfold-PPP values using `kfold_results.py`, including the following model:

    * `model3` : Simplest model (no factors) using a full covariance matrix

* The results of experiment of replacing 10% is in notebooks
`2.1.Muthen-women-experiment` and `2.2.Muthen-men-experiment`

* The results of using the no-u model is in
`3.1.Muthen-women-nou` and `3.2.Muthen-men-nou`

* The results of using the no-u model with 10% experiment is in  
`4.1.Muthen-women-nou-exp2`



## To Do

* ~~Experiment to detect extreme values of y's~~
* ~~Check calculations of PPP against Muthen paper~~
* ~~Add kfold PPP calculations~~
* Run model3 (unrestricted) on fabian - need to update the muthen_exp_kfold.py script first
* check with Kostas if `model-results.py` is computing PPP values correctly
