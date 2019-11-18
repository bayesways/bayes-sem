data {
  int<lower=1> N;
  int<lower=1> K;
  int<lower=1> J;
  int<lower=0, upper=1> DD[N, J];
}

transformed data{
  vector[J] zeros_J = rep_vector(0, J);
  vector[K] zeros_K = rep_vector(0, K);
  cov_matrix[J] I_J = diag_matrix(rep_vector(1, J));
  cov_matrix[K] I_K = diag_matrix(rep_vector(1, K));
  real<lower=0> c0 = 2.5;
  real<lower=0> c = 0.4;
}

parameters {
  vector[J] alpha;
  matrix[J,K] beta;
  matrix[N,K] zz;
  matrix<lower=0>[N,J] uu;
}

transformed parameters{
  matrix[N,J] yy;
  for (n in 1:N) yy[n,] = to_row_vector(alpha) + zz[n,] * beta' + uu[n,];
}

model {
  to_vector(beta) ~ normal(0, 1);
  to_vector(alpha) ~ normal(0, 10);
  to_vector(zz) ~ normal(0,1);
  to_vector(uu) ~ normal(0, c);
  for (j in 1:J) DD[, j] ~ bernoulli_logit(yy[, j]);
}
