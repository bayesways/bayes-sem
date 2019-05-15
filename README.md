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


## Results:
* To run the experiment of Muthen data `run_muthen_exp1.py`. Usually we run this
on the server, collect the results and visualize them locally with
`1.1.Muthen-women` or `1.1.Muthen-men`

## To Do

* Experiment to detect extreme values of y's
