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

* We run the Normal-Normal model with marginal likelihood formulas in notebook
`src/5.1.1-NN-EFA.ipynb`

* We run the Normal-Normal model with augmented likelihood formulas in notebook
`src/5.1.2-NN-EFA-aug.ipynb`. This runs with worst mixing that `5.1.1`

* We run the Normal-Normal model with marginal likelihood and hierarchical priors on
$\Sigma_u$ in notebook `src/5.1.3-NN-EFA-marg-hier.ipynb`

* We run the IRT model with marginal likelihood formulas in notebook
`src/5.2.1-IRT-EFA-marg.ipynb`

* We run the IRT model with augmented likelihood formulas in notebook
`src/5.2.2-IRT-EFA-aug.ipynb`

* We run the IRT model with augmented likelihood and hierarchical priors on
$\Sigma_u$ in notebook `src/5.2.3-IRT-EFA-hierarchical-prior.ipynb`

* In `src/6.1` we have a model for a simple IRT using the marginal likelihood.
The observable $D$ is distributed as bernoulli_logit of the latent
variable $y \sim N(\mu, \Omega)$ where $\Omega = \beta \beta' + cI$

* In `src/6.2` we have a model for the same simple IRT model above
using the augmented likelihood.
The observable $D$ is distributed as bernoulli_logit of the latent
variable $y = \mu + z \beta' + u$ where $z \sim N(0,1)$ and
$\epsilon \sim N(0, I)$.

* In `6.2.2` is the same as above except the data are simulated without adding
the noise elements u.


* In `6.2 LSAT_1` notebooks run the model with and without u's with 1 factor

* In `6.2 LSAT_2` notebooks run the model with and without u's with 2 factors

* `Results` notebook collects results for social stats talk.

* `7.1 IRT_1_LSAT` We ran the model replacing the last 100 points with random
coin flips and saw the change in u's.

* CFA for Normal-Normal model with augmented likelihood formulas in notebook
`src/8.1.1-NN-CFA.ipynb`

* CFA for Normal-Normal model with marginal likelihood formulas in notebook
`src/8.1.2-NN-CFA.ipynb`

* We run the IRT model with augmented likelihood and hierarchical priors on
$\Sigma_u$ in notebook `src/8.2.1-IRT-CFA-aug-hier.ipynb`


## To Do

* In sequential setting when sampling with the weights need to use real weights,
not log weights.
