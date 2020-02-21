data {
  int<lower=1> N;
  int<lower=1> K;
  int<lower=1> J;
  int<lower=0, upper=1> DD[N, J];
}

transformed data{
  cov_matrix[J] I_J = diag_matrix(rep_vector(1, J));
  vector[J] zeros_J = rep_vector(0, J);

}

parameters {
  vector[J] alpha;
  matrix[3,K] beta_free; // 3 free eleements per factor
  matrix[J-3,K] beta_zeros; // 3 zero elements per factor
  cov_matrix[J] Omega_cov;
  matrix[N,K] zz_tilde;
  matrix[N,J] uu;
  cholesky_factor_corr[K] L_Phi;

}

transformed parameters{
  matrix[J,K] beta;
  matrix[N,K] zz;
  matrix[N,J] yy;

  for(j in 1:J) {
    for (k in 1:K) beta[j,k] = 0;
  }
  // set the free elements
  for (k in 1:K) beta[1+3*(k-1) : 3+3*(k-1), k] = beta_free[1:3,k];
  // set the zero elements
  beta[4:J, 1] = beta_zeros[1:(J-3), 1];
  beta[1:(J-3), K] = beta_zeros[1:(J-3), K];
  
  for (n in 1:N) zz[n,] =  zz_tilde[n,] * L_Phi;
  for (n in 1:N) yy[n,] = to_row_vector(alpha) + zz[n,] * beta' + uu[n,];
}

model {
  to_vector(beta_free) ~ normal(0, 1);
  to_vector(beta_zeros) ~ normal(0, 0.1);
  to_vector(alpha) ~ normal(0, 10);
  to_vector(zz_tilde) ~ normal(0,1);
  L_Phi ~ lkj_corr_cholesky(2);
  Omega_cov ~ inv_wishart(J+6, I_J);
  for (n in 1:N) uu[n,] ~ multi_normal(zeros_J, Omega_cov);
  for (j in 1:J) DD[, j] ~ bernoulli_logit(yy[, j]);
}

generated quantities{
  matrix[J,J] betabeta =  beta * beta';
  corr_matrix[K] Phi_cov = multiply_lower_tri_self_transpose(L_Phi);
}

