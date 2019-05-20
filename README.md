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

    $ run_muthen_exp1.py <num_warmup> <num_samples> <num_chains> <men/women>

with optional flags for `-th` (`--task_handle`), `-pm` (`--print_model`), and
`-xdir` (`--existing_directory`). The results are saved in
`src/log/<date-time>_<task_handle>`.

If an existing directory is given then the script looks for an existing compiled
stan model to load and run with the new number of iterations.


* The same model but replacing 10% of the data with 10's `run_muthen_exp2.py`.
The command runs as follows

    $ run_muthen_exp2.py <num_warmup> <num_samples> <num_chains> <men/women>


* To use the model without u's add the flag `--use_u 0`, it works for both
 `run_muthen_exp1.py` and  `run_muthen_exp2.py`


## Results:
* To run the experiment of Muthen data `run_muthen_exp1.py`. Usually we run this
on the server, collect the results and visualize them locally with
`1.1.Muthen-women` or `1.2.Muthen-men`

* The results of experiment of replacing 10% is in notebooks
`2.1.Muthen-women-experiment` and `2.2.Muthen-men-experiment`


* The results of using the no-u model is in
`3.1.Muthen-women-nou` and `3.2.Muthen-men-nou`

* The results of using the no-u model with 10% experiment is in  
`4.1.Muthen-women-nou-exp2`

## To Do

* ~~Experiment to detect extreme values of y's~~
