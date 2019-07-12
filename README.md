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

    $ run_muthen_exp1.py <num_warmup> <num_samples> <num_chains> <men/women> <model_code>

with optional flags for `-th` (`--task_handle`), `-pm` (`--print_model`), and
`-xdir` (`--existing_directory`). The results are saved in
`src/log/<date-time>_<task_handle>`.

To choose which model to run use the `<model_code>` option as follows:
"0:full model, 1:no u's, 2: no u's no approx zero betas "

If an existing directory is given then the script looks for an existing compiled
stan model to load and run with the new number of iterations.


* The same model but replacing 10% of the data with 10's `run_muthen_exp2.py`.
The command runs as follows

    $ run_muthen_exp2.py <num_warmup> <num_samples> <num_chains> <men/women>


* Script `muthen_exp_ppp_kfold.py` runs a kfold version of the `run_muthen_exp1`
script. It works as follows

    $ run_muthen_exp1.py <num_warmup> <num_samples> <num_chains> <men/women> <model_code>

with optional flags for `-th` (`--task_handle`), `-pm` (`--print_model`), `-nfl`
(`n_splits`) and `-xdir` (`--existing_directory`). The results are saved in
`src/log/<date-time>_<task_handle>`.

To choose which model to run use the `<model_code>` option as follows:
"0:full model, 1:no u's, 2: no u's no approx zero betas "

If an existing directory is given then the script looks for an existing compiled
stan model to load and run with the new number of iterations.


## Results:

* To run the experiment of Muthen data `run_muthen_exp1.py`. Usually we run this
on the server, collect the results and visualize them locally with
`1.1.Muthen-women` or `1.2.Muthen-men`. Also see more recent results for
`women` only run on fabian at
`10.Muthen-results-model0`, `10.Muthen-results-model1`, and `10.Muthen-results-model2`
respectively for different models.

* The results of experiment of replacing 10% is in notebooks
`2.1.Muthen-women-experiment` and `2.2.Muthen-men-experiment`


* The results of using the no-u model is in
`3.1.Muthen-women-nou` and `3.2.Muthen-men-nou`

* The results of using the no-u model with 10% experiment is in  
`4.1.Muthen-women-nou-exp2`

## To Do

* ~~Experiment to detect extreme values of y's~~
* ~~Check calculations of PPP against Muthen paper~~
* Add kfold PPP calculations
