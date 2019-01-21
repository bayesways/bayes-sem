# Sequential Monte Carlo

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


## Software Use


### Particle filter

* Particle filter for the simple case of $y \sim N(\mu, \sigma)$. It uses the
conjugate prior formula. This way we can analytically calculate the marginal
likelihood, to check the accuracy of our approximation.
notebook `1.1 PF - Gibbs.ipynb`, script `exp.conj.py`. There
is also a script to check the accuracy of this conjugate sampler `test_mcmc.py`.

* Particle filter for the simple case of $y \sim N(\mu, \Sigma)$. It uses a
Gibbs kernel with independent priors.
notebook `1.2 PF - Gibbs (Multivariate).ipynb`, script `exp.gibbs.py`

* Particle filter for the simple case of $y \sim N(\mu, \sigma)$. It uses a
Metropolis-Hastings kernel with independent priors.
notebook `1.3 PF - MH.ipynb`, script `exp.mh.py`. For multivariate
case see notebook `1.4 Particle Filter - MH (Multivariate).ipynb`,
script `exp.mh_multivariate.py`.

### MCMC with Gibbs+MH

* MCMC for latent variable model $z \sim N(\mu, \Sigma)$ and $z_c = y_c$ but $y_b = \text{expit}(z_b)$. Parameters $\theta$ are Gibbs and $z$ using an Adaptive
Metropolis kernel. We've implemented 2 different kernels. See notebook `3.1` for
a description of the differences. In `3.1 MCMC - Gibbs_MH.ipynb` we have the
independent sampler and  `3.1_MCMC_Gibbs_MH.py` is the respective script. In
`3.1.2 MCMC - Gibbs_MH - Random Walk.ipynb` is the random walk version of the
kernel. See results with `Plot results - GibbsMH` is the result of running
`3.1` for 50 experiments saved in
 `src/log/201811...`

 In notebooks `3.2.* ` we have the respective multivariate cases for any number
 of variables.


### Pseudomarginal MH

* Pseudomarginal MCMC for latent variable model.
PsMCMC marginalizes out $z$ and updates parameters $\theta$ using an Adaptive
Metropolis kernel. notebook `2.1 Pseudomarginal MH.ipynb`. and `2.1_PsMH.py`.
See results with `Plot results - Pseudomarginal MH` saved in
 `src/log/201811...`.

 The notebook only works for J = 2, i.e. one continuous and one binary variable.
 This can be easily generalized but we have not done it yet. The main functions
 that impose the constrain are the `theta_transform` functions that map between
 the constrained parameter space and the unconstrained sampling space, aka the
 real line.

 * For the alternative computation see `2.2 Pseudomarginal MH.ipynb`, the
 codebase being at `psmh2.py`. However, it's still not tested to work properly.


### Pseudomarginal Gibbs

* Run the particle Gibbs algorithm using $p(z|\theta)$ as in PsMCMC and $p(\theta|z)$ as in Gibbs+MH. See notebook `4.1` and `4.2`. For a script see `4.1_Ps_Gibbs_MH.py`
script. See results with `Plot results - PsGibbs` saved in
 `src/log/201811....`


### SMC^2
* Run the SMC^2 algorithm using a Pseudmarginal Gibbs kernel. See notebook
`5.1 SMC2 - Gibbs(log_implementation)` notebook

* Adaptive Tempered SMC^2 is implemented in notebooks 6. And is run on the real
data in notebooks 7.

## ToDo:

* ~~Build multivariate version~~
* ~~Start organizing code~~
* ~~improve plotting for mu, Sigma, R~~
* To test that filter with conjugate priors works for 1d run experiment
for 50 repetitions (`src/exp.conj.py`) and make a box plot of the results.
See `log/exp.conj3`
* ~~To test that filter with Gibbs kernel works run experiment for 50 repetitions
(`src/exp.gibbs.py`) and make a box plot of the results with Gibbs. See
`log/exp.gibbs2`~~
* ~~Also test Multivariate version of Gibbs with independent priors (see `3.1 ipynb`)~~
* ~~Add kernel of MH and test that it works with the particles
filter~~ see results of 50 experiments `Plot Results - MH.ipynb`
* ~~Add kernel of MH (Multivariate) and test that it works with the particles
filter see results of 50 experiments - run `src/exp.mh.multivariate.py`~~
see results of 50 experiments `Plot Results - MH.ipynb`
* Test conjugate prior MCMC see `src/test_mcmc.py` and results in
`src/plot results.ipynb`
* ~~Make sure that the priors used for the particles and the MCMC kernel are the
same.~~
* ~~Add estimation using analytical formulas~~
* ~~Add Marginal likelihood estimation in code for 1d~~
* ~~Add pseudomarginal MCMC (PMCMC) 1, working only with $p(z|\theta)$~~
* ~~Check mistake in code for particle filter line 199 in resample function~~
* ~~In PMCMC1 add better proposal, perhaps learn it from a the standard scheme
of Gibbs+MH~~
* ~~Add pseudomarginal Gibbs, with $p(z|\theta)$ and $p(\theta|z)$~~
* Check efficiency of old Gibbs_MH code. It should be about 1000 times faster
than PsGibbs.
* Check if it's more efficient to propose and accept all z's together, instead
of one subject at a time.
* ~~Run PsGibbs on real data.~~
* ~~Check overflow notes on `5.1` notebook~~


## Questions:

* Running the experiment with conjugate priors seems to bias the mean towards 0.
Why is that? See `src/plot results.ipynb`, samples at `log/exp.conj2`
* With Gibbs it works, even if the particles and the kernel do not have exactly
the same prior. Why? See `src/plot results.ipynb`, samples at `log/exp.gibbs2`
* the particles \theta are ordered pairs (mu_1,sigma_1), ...., (mu_N, sigma_N)
where N is the number of particles I use, e.g. N = 1000. Correct? - Yes
* When I resample the particles I preserve the pairings. That means, for example,
 that if m_3 is sampled then necessarily sigma_3 is sample as well. Is that
 right? *-* Yes, the parameters are sampled jointly.
* When I jitter a particle \theta_i I run an MCMC scheme using \theta_i as initial
value. Does this MCMC scheme need to be MH or can I use Gibbs as well? Either.
* The update step $\omega^m <- \omega^m u_t(\theta^m)$ is equivalent to
$$\omega^m <- p(y_t | y_{1:t-1} , \theta^m) * p(y_{t-1} | y_{1:t-2} , \theta^m)$$
which combined with
$$p(y_{1:t} | \theta^m) =  p(y_t | y_{1:t-1} , \theta^m) \ldots  p(y_1|\theta^m)$$
means that $\omega^m$ is the value of the likelihood $L_t(\theta^m)$
at moment $t$. Is that right? What about the unit reset of weights for some steps?
*-* Yes, that's right.
* How to construct final distributions using weights & particles? By density
estimation of samples from particles? *-* Yes, the weights are important for
estimation. If in the last step you jitter, and hence make all weights equal
to 1, you don't have to worry about that. Both methods are correct.
* (See `ipynb 1.`) To find the expectation of any quantity of interest $f(\theta)$
$$E(f(\theta) | y_{1:t}) = \frac{ \sum \omega^m f(\theta^m) }{ \sum \omega^m}$$
hence
$$E(\theta | y_{1:t}) = \frac{ \sum \omega^m \theta^m }{ \sum \omega^m}$$.
How to find $Var(\theta | y_{1:t}) = E((\theta - \theta_0)^2 | y_{1:t})$ ? *-* Correct.
For variance use the formula $Var(X) = E(X^2) - E(X)^2$.
