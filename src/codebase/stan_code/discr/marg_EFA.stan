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
  matrix[N,J] yy;
  vector[J] mu;
  vector[J] uu[N];
  cov_matrix[J] Sigma_u;
  matrix[J,K] beta;
}

transformed parameters{
  cov_matrix[J] Omega = beta * beta' + I_c;
}

model {
  to_row_vector(beta) ~ normal(0, 1);
  to_row_vector(mu) ~ normal(0, 1);
  Sigma_u ~ inv_wishart(J+6, I_c);
  for (n in 1:N) uu[n,] ~ multi_normal(zeros, Sigma_u);
  for (n in 1:N) yy[n,] ~ multi_normal(mu+uu[n,], Omega);
  for (j in 1:J) DD[, j] ~ bernoulli_logit(yy[, j]);
}

generated quantities{
  matrix[J,J] Omega_beta = beta * beta';
}

