data {
  int<lower=1> N;
  int<lower=1> K;
  int<lower=1> J;
  matrix[N,J] yy;
  vector[J] sigma_prior;
}

transformed data{
  vector[J] zeros = rep_vector(0, J);
  cov_matrix[J] I = diag_matrix(rep_vector(1, J));
  real<lower=0> c0 = 2.5;
}

parameters {
  vector<lower=0>[J] sigma_square;
  matrix[2,K] beta_free; // 2 free eleements per factor
  matrix[J-3,K] beta_zeros; // 3 zero elements per factor
  cholesky_factor_corr[K] Phi_corr_chol;
  matrix[N,J] uu;
  cov_matrix[J] Omega;
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

  // set the zero elements
  beta[4:J, 1] = beta_zeros[1:(J-3), 1];
  for (k in 2:(K-1)) {
    beta[1:3*(k-1), k] = beta_zeros[1:3*(k-1), k];
    beta[4+3*(k-1):J, k] = beta_zeros[3*(k-1)+1:J-3, k];
  }
  beta[1:(J-3), K] = beta_zeros[1:(J-3), K];

  Marg_cov = beta * Phi_cov * beta'+ Theta;

}

model {
  to_vector(beta_free) ~ normal(0, 1);
  to_vector(beta_zeros) ~ normal(0, 0.1);
  for(j in 1:J) sigma_square[j] ~ inv_gamma(c0, (c0-1)/sigma_prior[j]);
  Phi_corr_chol ~ lkj_corr_cholesky(2);
  Omega ~ inv_wishart(J+6, I);
  for (n in 1:N){
    to_vector(uu[n,]) ~ multi_normal(zeros, Omega);
  }
  for (n in 1:N){
    yy[n, ] ~ multi_normal(to_vector(uu[n,]),  Marg_cov);
  }
  
}

generated quantities{
  matrix [K, K] V_corr = multiply_lower_tri_self_transpose(Phi_corr_chol);
  matrix [J, J] Marg_cov2 = Marg_cov + Omega ;
  vector<lower=0>[J] sigma = sqrt(sigma_square);
}
