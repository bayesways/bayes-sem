data {
  int<lower=1> N;
  int<lower=1> K;
  int<lower=1> J;
  int<lower=0, upper=1> DD[N, J];

}

transformed data{
  real<lower=0> c = 0.01;
  vector[J] zeros = rep_vector(0, J);
  cov_matrix[J] I_c = diag_matrix(rep_vector(c, J));
}

parameters {
  vector[J] alpha;
  matrix[K,K] beta_t;
  cholesky_factor_cov[K] V_chol;
  matrix[N,K] zz;
  matrix[N,J] uu;
  cov_matrix[J] Sigma_u;
}

transformed parameters{
  matrix[J,K] beta;
  matrix[N,J] yy;
  for (j in 1:J){
    for (k in 1:K) beta[j,k] = 0;
  }
  beta[1,1] = 1;
  beta[2:3,1] = beta_t[1:2,1];
  beta[4,2] = 1;
  beta[5:6,2] = beta_t[1:2,2];
  for (n in 1:N){
    yy[n,] = to_row_vector(alpha) + zz[n,] * beta' + uu[n,];
  }
  
}

model {
  to_vector(beta_t) ~ normal(0, 1);
  to_vector(alpha) ~ normal(0, 1);
  V_chol ~ lkj_corr_cholesky(2);
  for (n in 1:N){
      to_vector(zz[n, ]) ~ multi_normal_cholesky(rep_vector(0, K), V_chol);
  }
  Sigma_u ~ inv_wishart(J+6, I_c);
  for (n in 1:N){
    to_vector(uu[n,]) ~ multi_normal(zeros, Sigma_u);
  }
  for (j in 1:J) DD[, j] ~ bernoulli_logit(yy[, j]);

}

generated quantities{
  matrix [J, J] Omega_beta = beta * V_chol * V_chol' * beta';
  matrix [K, K] V = V_chol * V_chol';
}

