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

* Gibbs + Pseudomarginal in `2. Pseudomarginal` notebook. Here the sampling
conditionals are the same as above except $p(z| y,\theta)$. We generate many
z_i for each row and choose from a multinomial. Was initiated at the correct
values.

* Gibbs + Pseudomarginal in `2.2 Pseudomarginal` notebook. Here the sampling
conditionals are the same as above except $p(\beta| y, \sigma)$.

* SMC2 + Gibbs in `3.1 SMC2` notebook. The weights are calculated using the marginal
likelihood (not the augmented likelihood). The jittering MCMC is full Gibbs. For
an alternative weight calculation using the augmented likelihood, marginalizing
over $Z$ see `smc2_auglklh`.

* HMC sampler in `4. HMC` where $z$ is sampled post-HMC using the eigenvector
of $\beta \beta^T$


## To Do

* When sampling with the weights need to use real weights, not log weights.
