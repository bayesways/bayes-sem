data {
  int<lower=1> N;
  int<lower=1> K;
  int<lower=1> J;
  int<lower=0, upper=1> DD[N, J];
}

transformed data{
  cov_matrix[J] I_J = diag_matrix(rep_vector(1, J));
  real<lower=0> c = 0.7;
}

parameters {
  vector[J] alpha;
  matrix[J,K] beta;
  matrix[N,J] yy;
}

transformed parameters{
  cov_matrix[J] Marg_cov;
  cov_matrix[J] Omega_cov;
  Omega_cov = diag_matrix(rep_vector(square(c), J));
  Marg_cov = beta * beta'+ Omega_cov;

}

model {
  to_vector(beta) ~ normal(0, 1);
  to_vector(alpha) ~ normal(0, 10);
  for (n in 1:N) yy[n,] ~ multi_normal(alpha, Marg_cov);
  for (j in 1:J) DD[, j] ~ bernoulli_logit(yy[, j]);
}

generated quantities{
  matrix[J,J] betabeta =  beta * beta';
}
