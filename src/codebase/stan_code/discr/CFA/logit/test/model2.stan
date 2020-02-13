data {
  int<lower=1> N;
  int<lower=1> K;
  int<lower=1> J;
  matrix[N,J] DD;

}

transformed data{
  vector[K] zeros_K = rep_vector(0, K);
  vector[J] zeros_J = rep_vector(0, J);
  cov_matrix[J] I_J = diag_matrix(rep_vector(1, J));
}

parameters {
  vector[J] alpha;
  matrix[3,K] beta_free; // 3 free eleements per factor
  matrix[J-3,K] beta_zeros; // 3 zero elements per factor
  cholesky_factor_corr[K] L_Phi;
  cov_matrix[J] Omega_cov;
  matrix[N,J] uu;
  
}

transformed parameters{
  matrix[J,K] beta;
  cov_matrix[J] Marg_cov;
  corr_matrix[K] Phi_cov;
  
  for(j in 1:J) {
    for (k in 1:K) beta[j,k] = 0;
  }
  // set the free elements
  for (k in 1:K) beta[1+3*(k-1) : 3+3*(k-1), k] = beta_free[1:3,k];
  // set the zero elements
  beta[4:J, 1] = beta_zeros[1:(J-3), 1];
  beta[1:(J-3), K] = beta_zeros[1:(J-3), K];
  
  
  Phi_cov = multiply_lower_tri_self_transpose(L_Phi);
  Marg_cov = beta * Phi_cov * beta'+ I_J;

}

model {
  to_vector(beta_free) ~ normal(0, 1);
  to_vector(beta_zeros) ~ normal(0, 0.1);
  to_vector(alpha) ~ normal(0, 10);
  L_Phi ~ lkj_corr_cholesky(2);
  Omega_cov ~ inv_wishart(J+4, I_J);
  for (n in 1:N) to_vector(uu[n,]) ~ multi_normal(zeros_J, Omega_cov);
  // for (j in 1:J) DD[, j] ~ bernoulli_logit(yy[, j]);
  for (n in 1:N) DD[n,] ~ multi_normal(to_row_vector(alpha) + uu[n,], Marg_cov);
}

