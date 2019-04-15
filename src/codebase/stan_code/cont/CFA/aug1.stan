data {
  int<lower=1> N;
  int<lower=1> K;
  int<lower=1> J;
  matrix[N,J] yy;
}

parameters {
  vector<lower=0>[J] sigma;
  vector[J] alpha;
  matrix[K,K] beta_t;
  cholesky_factor_cov[K] V_chol;
  matrix[N,K] zz;

}

transformed parameters{
  matrix[J,K] beta;
  cov_matrix[J] Sigma_epsilon;
  matrix[N,J] mean_yy;
  Sigma_epsilon = diag_matrix(square(sigma));
  for (j in 1:J){
    for (k in 1:K) beta[j,k] = 0;
  }
  beta[1,1] = 1;
  beta[2:3,1] = beta_t[1:2,1];
  beta[4,2] = 1;
  beta[5:6,2] = beta_t[1:2,2];
  
  for (n in 1:N){
    mean_yy[n,] = to_row_vector(alpha) + zz[n,] * V_chol' * beta';
  }
  
}

model {
  to_vector(beta_t) ~ normal(0, 1);
  to_vector(alpha) ~ normal(0, 1);
  V_chol ~ lkj_corr_cholesky(2);
  for (n in 1:N){
      to_vector(zz[n, ]) ~ multi_normal_cholesky(rep_vector(0, K), V_chol);
  }

  for (n in 1:N){
    yy[n, ] ~ multi_normal(mean_yy[n,],  Sigma_epsilon);
  }
}

generated quantities{
  matrix [J, J] Omega_beta = beta * V_chol * V_chol' * beta';
}
