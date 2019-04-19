data {
  int<lower=1> N;
  int<lower=1> K;
  int<lower=1> J;
  matrix[N,J] yy;
}

transformed data{
  real<lower=0> c = 1;
  vector[J] zeros = rep_vector(0, J);
  cov_matrix[J] I_c = diag_matrix(rep_vector(c, J));
}

parameters {
  vector<lower=0>[J] sigma;
  vector<lower=0>[K] sigma_z;
  vector[J] alpha;
  matrix[2, K] beta_free;
  matrix[J-3, K] beta_zeros;
  cholesky_factor_corr[K] V_corr_chol;
  matrix[N,K] zz;
  matrix[N,J] uu;
  cov_matrix[J] Sigma_u;
}

transformed parameters{
  matrix[J,K] beta;
  cov_matrix[J] Sigma_epsilon;
  matrix[N,J] mean_yy;
  Sigma_epsilon = diag_matrix(square(sigma));
  for (j in 1:J){
    for (k in 1:K) beta[j,k] = 0;
  }
  
  // set ones 
  for (k in 1:K) beta[1+3*(k-1), k] = 1;
  
  // set the free elements
  for (k in 1:K) beta[2+3*(k-1) : 3+3*(k-1), k] = beta_free[1:2,k];
  
  // set the zero elements
  beta[4:J, 1] = beta_zeros[1:(J-3), 1];
  for (k in 2:(K-1)) beta[1:3*(k-1), k] = beta_zeros[1:3*(k-1), k];
  beta[1:(J-3), K] = beta_zeros[1:(J-3), K];
  
  for (n in 1:N){
    mean_yy[n,] = to_row_vector(alpha) + zz[n,] * beta' + uu[n, ];
  }
}

model {
  to_vector(beta_free) ~ normal(0, 1);
  to_vector(beta_zeros) ~ normal(0, 0.1);
  to_vector(alpha) ~ normal(0, 1);
  sigma ~ cauchy(0,2);
  sigma_z ~ cauchy(0,2);
  V_corr_chol ~ lkj_corr_cholesky(2);
  for (n in 1:N){
      to_vector(zz[n, ]) ~ multi_normal_cholesky(rep_vector(0, K),
      diag_pre_multiply(sigma_z, V_corr_chol));
  }
  Sigma_u ~ inv_wishart(J+6, I_c);
  for (n in 1:N){
    to_vector(uu[n,]) ~ multi_normal(zeros, Sigma_u);
  }
  for (n in 1:N){
    yy[n, ] ~ multi_normal(mean_yy[n,],  Sigma_epsilon);
  }
}

generated quantities{
  matrix [K, K] V = multiply_lower_tri_self_transpose(diag_pre_multiply(sigma_z, V_corr_chol));
  matrix [J, J] Omega_beta = beta * V * beta';
}

