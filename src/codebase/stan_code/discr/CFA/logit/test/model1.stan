data {
  int<lower=1> N;
  int<lower=1> J;
  int<lower=0, upper=1> DD[N, J];

}

transformed data{
  vector[J] zeros_J = rep_vector(0, J);
  // cov_matrix[J] I_J = diag_matrix(rep_vector(1, J));
}

parameters {
  cholesky_factor_corr[J] L;
  vector<lower=0>[J] sigma;
  matrix[N,J] yy;
}

// transformed parameters{
// }

model {
  for (n in 1:N) yy[n,] ~ multi_normal_cholesky(zeros_J, diag_pre_multiply(sigma, L));
  for (j in 1:J) DD[, j] ~ bernoulli_logit(yy[, j]);
}

generated quantities{
  corr_matrix[J] Omega_corr = multiply_lower_tri_self_transpose(L);
  cov_matrix[J] Omega_cov = multiply_lower_tri_self_transpose( diag_pre_multiply(sigma,L));
}
