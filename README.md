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

* HMC sampler in `src/hmc/1. hmc` where $z$ is sampled post-HMC using the eigenvector
of $\beta \beta^T$.

* In `src/irt/1. hmc.ipynb` we have a model for a simple IRT model with `mu=0`
using the marginal likelihood and `src/irt/2. hmc.ipynb` we have a model for
`mu \neq 0`. The observable $D$ is distributed as bernoulli_logit of the latent
variable $y \sim N(\mu, \Omega)$ where $\Omega = \beta \beta' + I$

* In `src/irt/3. hmc.ipynb` we have a model for the same simple IRT model above
using the augmented likelihood.
The observable $D$ is distributed as bernoulli_logit of the latent
variable $y = \mu + z \beta' + \epsilon$ where $z \sim N(0,1)$ and
$\epsilon \sim N(0, I)$.

* We do the Normal Normal factor model with $u$ in `src/5 Normal normal.ipynb`

* We do the Logit Normal IRT factor model with $u$ for binary data in 
`src/6 IRT.ipynb`

## To Do

* When sampling with the weights need to use real weights, not log weights.
