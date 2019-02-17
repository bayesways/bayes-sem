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


## How to use:

* Gibbs sampler in `1. Gibbs notebook`

* Gibbs + Pseudomarginal in `2.1 Pseudomarginal` notebook. Here the sampling
conditionals are the same as above except $p(z| y,\theta)$.

* Gibbs + Pseudomarginal in `2.2 Pseudomarginal` notebook. Here the sampling
conditionals are the same as above except $p(\beta| y, \sigma)$.

* SMC2 + Gibbs in `3.1 SMC2` notebook. The weights are calculated using the
augmented likelihood, marginalizing over $Z$. The jittering MCMC is full Gibbs.

* SMC2 + Gibbs in `3.2 SMC2` notebook. The weights are calculated using the marginal
likelihood (not the augmented likelihood). The jittering MCMC is full Gibbs.
