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
  vector[2] beta_free1; // 1st factor
  vector[3] beta_free2; // 2nd factor (enforce beta lower triangular)
  vector[3] beta_zeros1; 
  vector[2] beta_zeros2; 
  cov_matrix[K] Phi_cov;
  matrix[N,K] zz;
}

transformed parameters{
  matrix[J,K] beta;
  matrix[N,J] yy;

  beta[1,1] = 1;
  beta[2:3, 1] = beta_free1;
  beta[4:J, 1] = beta_zeros1;

  beta[1,2] = beta_free2[1]; // cross loading of first variable to both factors
  beta[2:3,2] = beta_zeros2;
  beta[4, 2] = 1;
  beta[5:J, 2] = beta_free2[2:3];

  for (n in 1:N) yy[n,] = to_row_vector(alpha) + zz[n,] * beta';
}

model {
  to_vector(beta_free1) ~ normal(0, 1);
  to_vector(beta_free2) ~ normal(0, 1);
  to_vector(beta_zeros1) ~ normal(0, 0.1);
  to_vector(beta_zeros2) ~ normal(0, 0.1);
  to_vector(alpha) ~ normal(0, 10);
  Phi_cov ~ inv_wishart(J+2, I_K);
  for (n in 1:N) to_vector(zz[n,])  ~ multi_normal_cholesky(zeros_K, Phi_cov);
  for (j in 1:J) DD[, j] ~ bernoulli_logit(yy[, j]);
}

generated quantities{
  matrix[J,J] betabeta =  beta * beta';
}

