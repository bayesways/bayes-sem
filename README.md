# Factor Analysis

## Setup

Initial install:

    $ conda env create -f environment.yml
    $ source activate smc

Update:

    $ conda env update -f environment.yml

Deactivate Environment:

    $ source deactivate

Remove:

    $ conda remove -n smc --all


Convert `a.ipynb` to `a.py`:

  $ jupyter nbconvert --to script a.ipynb
