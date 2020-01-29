data {
  int<lower=1> N;
  int<lower=1> K;
  int<lower=1> J;
  int<lower=0, upper=1> DD[N, J];
}

transformed data{
  vector[K] zeros_K = rep_vector(0, K);
  vector[J] zeros_J = rep_vector(0, J);
}

parameters {
  vector[J] alpha;
  matrix[3,K] beta_free; // 3 free eleements per factor
  matrix[J-3,K] beta_zeros; // 3 zero elements per factor
  cholesky_factor_corr[K] L_Phi;
  real<lower=0> c;
  cholesky_factor_corr[J] L_Omega;
  matrix[N,K] zz;
  matrix[N,J] uu_tilde;
}

transformed parameters{
  matrix[J,K] beta;
  matrix[N,J] yy;
  matrix[N,J] uu;
  vector<lower=0>[J] sigma_u = rep_vector(c, J);

  for(j in 1:J) {
    for (k in 1:K) beta[j,k] = 0;
  }
  // set the free elements
  for (k in 1:K) beta[1+3*(k-1) : 3+3*(k-1), k] = beta_free[1:3,k];
  // set the zero elements
  beta[4:J, 1] = beta_zeros[1:(J-3), 1];
  beta[1:(J-3), K] = beta_zeros[1:(J-3), K];
  
  for (n in 1:N) uu[n,] = uu_tilde[n,] * diag_pre_multiply(sigma_u, L_Omega);

  for (n in 1:N) yy[n,] = to_row_vector(alpha) + zz[n,] * beta' + uu[n,];
}

model {
  to_vector(beta_free) ~ normal(0, 1);
  to_vector(beta_zeros) ~ normal(0, 0.1);
  to_vector(alpha) ~ normal(0, 10);
  L_Phi ~ lkj_corr_cholesky(2);
  to_vector(uu_tilde) ~ normal(0, 1);
  c ~ cauchy(0,2.5);
  L_Omega ~ lkj_corr_cholesky(10);
  for (n in 1:N) to_vector(zz[n,])  ~ multi_normal_cholesky(zeros_K, L_Phi);
  for (j in 1:J) DD[, j] ~ bernoulli_logit(yy[, j]);
}

generated quantities{
  corr_matrix[K] Phi_cov = multiply_lower_tri_self_transpose(L_Phi);
  cov_matrix[J] Omega_cov = multiply_lower_tri_self_transpose(diag_pre_multiply(sigma_u, L_Omega));
  corr_matrix[J] Omega_corr = multiply_lower_tri_self_transpose(L_Omega);

}
