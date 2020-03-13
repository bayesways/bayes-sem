data {
  int<lower=1> N;
  int<lower=1> K;
  int<lower=1> J;
  int<lower=0, upper=1> DD[N, J];
}

transformed data{
  cov_matrix[J] I_J = diag_matrix(rep_vector(1, J));
  vector<lower=0>[J] sigma_omega;
  sigma_omega = rep_vector(0.7, J);
}

parameters {
  vector[J] alpha;
  matrix[J,K] beta;
  matrix[N,J] yy;
  cholesky_factor_corr[J] L_Omega;
}

transformed parameters{
  cov_matrix[J] Marg_cov;
  cov_matrix[J] Omega_cov;

  Omega_cov = multiply_lower_tri_self_transpose(diag_pre_multiply(sigma_omega, L_Omega));
  Marg_cov = beta * beta'+ Omega_cov;
}

model {
  to_vector(beta) ~ normal(0, 1);
  to_vector(alpha) ~ normal(0, 10);
  L_Omega ~ lkj_corr_cholesky(5);
  for (n in 1:N) yy[n,] ~ multi_normal(alpha, Marg_cov);
  for (j in 1:J) DD[, j] ~ bernoulli_logit(yy[, j]);
}

generated quantities{
  corr_matrix[J] Omega_corr = multiply_lower_tri_self_transpose(L_Omega);
  matrix[J,J] betabeta =  beta * beta';
}
