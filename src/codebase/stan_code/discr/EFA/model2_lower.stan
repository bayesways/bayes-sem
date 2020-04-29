data {
  int<lower=1> N;
  int<lower=1> K;
  int<lower=1> J;
  int<lower=0, upper=1> DD[N, J];
}

transformed data{
  cov_matrix[J] I_J = diag_matrix(rep_vector(1, J));
}

parameters {
  vector[J] alpha;
  vector[J] beta_free1; // 1st factor
  vector[J-1] beta_free2; // 2nd factor (enforce beta lower triangular)
  matrix[N,J] yy;
  cov_matrix[J] Omega_cov;
}

transformed parameters{
  matrix[J,K] beta;
  cov_matrix[J] Marg_cov;
  beta[1,2] = 0;
  beta[1:J, 1] = beta_free1;
  beta[2:J, 2] = beta_free2;
  Marg_cov = beta * beta'+ Omega_cov;
}

model {
  to_vector(beta_free1) ~ normal(0, 1);
  to_vector(beta_free2) ~ normal(0, 1);
  to_vector(alpha) ~ normal(0, 10);
  Omega_cov ~ inv_wishart(J+6, I_J);
  for (n in 1:N) yy[n,] ~ multi_normal(alpha, Marg_cov);
  for (j in 1:J) DD[, j] ~ bernoulli_logit(yy[, j]);
}

generated quantities{
  matrix[J,J] betabeta =  beta * beta';
}
