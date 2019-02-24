library(tidyverse)
library(rstan)
library(bayesplot)
library(gtools)
library(MASS)
library(ggplot2)
color_scheme_set('red')
options(mc.cores = parallel::detectCores())



set.seed(6823234)
N <-100
J <- 6
K <- 2

beta <- matrix(c(1,2,-2,3,-1,-1,0,1,3,-1,1,-1), ncol=K, nrow=J)
sigma<- runif(J, 0, 3)
Sigma<-diag(sigma)
Omega <- beta %*% t(beta) + Sigma

zz <- mvrnorm(n = N, mu =  rep(0,K), Sigma = diag(K)) # latent continuous variables
ee <- mvrnorm(n = N, mu =  rep(0,J), Sigma = Sigma) # error terms

yy <- zz %*% t(beta) + ee


### Plotting
yplot<- data.frame(yy)
colnames(yplot) <- c('y1','y2','y3','y4','y5','y6')
yplot<- yplot %>% gather(variable, value)

ggplot(yplot, aes(value, fill = variable, colour = variable)) +
  geom_density(alpha = 0.2)+
  scale_x_continuous(breaks = seq(0,100, 10))


list(N = N, K = K, J = J, yy = yy)

model<- stan_model(file='model_code.stan')

### Fiting in Stan
fit <- sampling(model,
            data=list(N = N, K = K, J = J, yy = yy),
            iter=1000,
            chains = 4,
            seed=4938483)


stan_rdump(c("N", "K","J" , "yy" ), file="data.R")
print(fit, pars=c('beta', 'sigma'))


posterior <- as.array(fit)
dimnames(posterior)
mcmc_trace(posterior, pars = c("sigma"))

