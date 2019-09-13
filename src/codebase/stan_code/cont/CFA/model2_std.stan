data {
  int<lower=1> N;
  int<lower=1> K;
  int<lower=1> J;
  matrix[N,J] yy;
  vector[J] sigma_prior;
}

transformed data{
  real<lower=0> c0 = 2.5;
  vector[J] zeros = rep_vector(0, J);
}

parameters {
  vector<lower=0>[J] sigma_square;
  matrix[2,K] beta_free; // 2 free eleements per factor
  cholesky_factor_corr[K] Phi_corr_chol;
}

transformed parameters{
  cov_matrix[J] Theta;
  matrix[J,K] beta;
  cov_matrix [K] Phi_cov ;
  cov_matrix[J] Marg_cov;

  Theta = diag_matrix(sigma_square);
  Phi_cov = multiply_lower_tri_self_transpose(Phi_corr_chol);

  for(j in 1:J) {
    for (k in 1:K) beta[j,k] = 0;
  }

  // set ones
  for (k in 1:K) beta[1+3*(k-1), k] = 1;

  // set the free elements
  for (k in 1:K) beta[2+3*(k-1) : 3+3*(k-1), k] = beta_free[1:2,k];

  Marg_cov = beta * Phi_cov * beta'+ Theta;
}

model {
  to_vector(beta_free) ~ normal(0, 1);
  for(j in 1:J) sigma_square[j] ~ inv_gamma(c0, (c0-1)/sigma_prior[j]);
  Phi_corr_chol ~ lkj_corr_cholesky(2);
  for (n in 1:N){
    yy[n, ] ~ multi_normal(zeros, Marg_cov);
  }
}

generated quantities{
  matrix [K, K] Phi_corr = multiply_lower_tri_self_transpose(Phi_corr_chol);
  vector<lower=0>[J] sigma = sqrt(sigma_square);

}
