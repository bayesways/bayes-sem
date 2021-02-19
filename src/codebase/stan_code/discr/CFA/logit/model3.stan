data {
  int<lower=1> N;
  int<lower=1> K;
  int<lower=1> J;
  int<lower=0, upper=1> DD[N, J];
}

transformed data{
  vector[K] zeros_K = rep_vector(0, K);
  cov_matrix[K] I_K = diag_matrix(rep_vector(1, K));
}

parameters {
  vector[J] alpha;
  vector[3] beta_free1; // 1st factor
  vector[4] beta_free2; // 2nd factor 
  vector[3] beta_zeros1; 
  vector[2] beta_zeros2; 
  cholesky_factor_corr[K] L_Phi;
  matrix[N,K] zz;
}

transformed parameters{
  matrix[J,K] beta;
  matrix[N,J] yy;

  beta[1:3, 1] = beta_free1;
  beta[4:J, 1] = beta_zeros1;

  beta[1,2] = beta_free2[1]; // cross loading of first variable to both factors
  beta[2:3,2] = beta_zeros2;
  beta[4:J, 2] = beta_free2[2:4];

  for (n in 1:N) yy[n,] = to_row_vector(alpha) + zz[n,] * beta';
}

model {
  to_vector(beta_free1) ~ normal(0, 1);
  to_vector(beta_free2) ~ normal(0, 1);
  to_vector(beta_zeros1) ~ normal(0, 0.1);
  to_vector(beta_zeros2) ~ normal(0, 0.1);
  to_vector(alpha) ~ normal(0, 10);
  L_Phi ~ lkj_corr_cholesky(2);
  for (n in 1:N) to_vector(zz[n,])  ~ multi_normal_cholesky(zeros_K, L_Phi);
  for (j in 1:J) DD[, j] ~ bernoulli_logit(yy[, j]);
}

generated quantities{
  corr_matrix[K] Phi_cov;
  matrix[J,J] betabeta;
  betabeta =  beta * beta';
  Phi_cov = multiply_lower_tri_self_transpose(L_Phi);
}

