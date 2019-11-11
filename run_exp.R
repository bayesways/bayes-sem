library(rstan)

data<- read.csv("dat/clean_iri_data.csv")
J <- dim(data)[2] - 2
DD <- data[,3:dim(data)[2]]
stan_dat <- list(N = dim(data)[1], 
                 J = J, 
                 K = 1,
                 DD = DD)
fit <- stan(file='src/model_1factor.stan',
            data = stan_dat,
            # control = list('adapt_delta' = 0.9, 'max_treedepth' = 10),
            iter=10 ,
            warmup = 5,
            chains = 1,
            seed=4)

saveRDS(fit, "fit.rds" )