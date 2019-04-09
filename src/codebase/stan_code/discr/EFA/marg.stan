data {
  int<lower=1> N;
  int<lower=1> K;
  int<lower=1> J;
  int<lower=0, upper=1> DD[N, J];
}

transformed data{
  real<lower=0> c = 0.01;
  vector[J] zeros = rep_vector(0, J);
  cov_matrix[J] I_c = diag_matrix(rep_vector(c, J));
}

parameters {
  vector[J] alpha;
  matrix[J,K] beta;
  matrix[N,J] yy;
}

transformed parameters{
  cov_matrix[J] Omega = beta * beta'+ I_c;
}

model {
  to_row_vector(beta) ~ normal(0, 1);
  to_row_vector(alpha) ~ normal(0, 1);
  for (n in 1:N) yy[n, ] ~ multi_normal(alpha, Omega);
  for (j in 1:J) DD[, j] ~ bernoulli_logit(yy[, j]);
}

generated quantities{
  matrix [J, J] Omega_beta = beta * beta';
}

