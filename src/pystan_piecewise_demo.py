from pystan import StanModel
from pystan.constants import MAX_UINT
import numpy as np

# bernoulli model
model_code = """
    data {
      int<lower=0> N;
      int<lower=0,upper=1> y[N];
    }
    parameters {
      real<lower=0,upper=1> theta;
    }
    model {
      theta ~ beta(0.5, 0.5);  // Jeffreys' prior
      for (n in 1:N)
        y[n] ~ bernoulli(theta);
    }
"""

data = dict(N=10, y=[0, 1, 0, 1, 0, 1, 0, 1, 1, 1])
sm = StanModel(model_code=model_code)
# initial seed can also be chosen by user
# MAX_UINT = 2147483647
seed = np.random.randint(0, MAX_UINT, size=1)
fit = sm.sampling(data=data, seed=seed)

# reuse tuned parameters
stepsize = fit.get_stepsize()
# by default .get_inv_metric returns a list
inv_metric = fit.get_inv_metric(as_dict=True)
init = fit.get_last_position()

# increment seed by 1
seed2 = seed + 1

control = {"stepsize" : stepsize,
           "inv_metric" : inv_metric,
           "adapt_engaged" : False
           }
fit2 = sm.sampling(data=data,
                   warmup=0,
                   iter=1000,
                   control=control,
                   init=init,
                   seed=seed2)
