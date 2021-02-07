library("scoringRules")
obs <- rnorm(10)
crps(obs, "norm", mean = c(1:10), sd = c(1:10))
crps_norm(obs, mean = c(1:10), sd = c(1:10))


obs_n <- c(0, 1, 2)
sample_nm <- matrix(rep(rnorm(1e6, mean = 2, sd = 3),3), nrow = 3)
crps_sample(obs_n, dat = sample_nm)


sample_nm <- matrix(rnorm(1e6*3, mean = 2, sd = 3), nrow = 3)
crps_sample(obs_n, dat = sample_nm)


## ----Data-MCMC-example----------------------------------------------
data("gdp", package = "scoringRules")
data_train <- subset(gdp, vint == "2014Q1")
data_eval <- subset(gdp, vint == "2015Q1" & grepl("2014", dt))

## ----Sampling-MCMC-forecast-parameters------------------------------
h <- 4
m <- 20000
fc_params <- ar_ms(data_train$val, forecast_periods = h, n_rep = m)

## ----Regularize-forecast-parameter-data-format----------------------
mu <- t(fc_params$fcMeans)
Sd <- t(fc_params$fcSds)

## ----Sampling-ensemble-forecast-from-MCMC-forecast------------------
X <- matrix(rnorm(h * m, mean = mu, sd = Sd), nrow = h, ncol = m)

obs = c(0,1,2,3)
es_sample(obs, dat = X)

dim(X)
