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
* To run the experiment of Muthen data `run_muthen_exp1.py`. Usually we run this
on the server, collect the results and visualize them locally with
`1.1.Muthen-women` or `1.1.Muthen-men`

## To Do

* Experiment to detect extreme values of y's
