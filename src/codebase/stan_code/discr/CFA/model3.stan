data {
  int<lower=1> N;
  int<lower=1> K;
  int<lower=1> J;
  int<lower=0, upper=1> DD[N, J];
}

transformed data{
  vector[K] zeros_K = rep_vector(0, K);
  vector[J] zeros_J = rep_vector(0, J);
  cov_matrix[J] I_J = diag_matrix(rep_vector(1, J));
  cov_matrix[K] I_K = diag_matrix(rep_vector(1, K));
  real<lower=0> c0 = 2.5;
}

parameters {
  vector<lower=0>[J] sigma_square;
  vector[J] alpha;
  matrix[2,K] beta_free; // 2 free eleements per factor
  matrix[J-3,K] beta_zeros; // 3 zero elements per factor
  cov_matrix [K] Phi_cov;
  matrix[N,K] zz;
}

transformed parameters{
  matrix[J,K] beta;
  matrix[N,J] yy;
  
  for(j in 1:J) {
    for (k in 1:K) beta[j,k] = 0;
  }
  // set ones
  for (k in 1:K) beta[1+3*(k-1), k] = 1;
  // set the free elements
  for (k in 1:K) beta[2+3*(k-1) : 3+3*(k-1), k] = beta_free[1:2,k];
  // set the zero elements
  beta[4:J, 1] = beta_zeros[1:(J-3), 1];
  beta[1:(J-3), K] = beta_zeros[1:(J-3), K];

  for (n in 1:N) yy[n,] = to_row_vector(alpha) + zz[n,] * beta';
}

model {
  to_vector(beta_free) ~ normal(0, 1);
  to_vector(beta_zeros) ~ normal(0, 0.1);
  to_vector(alpha) ~ normal(0, 10);
  Phi_cov ~ inv_wishart(J+4, I_K);
  for (n in 1:N) to_vector(zz[n,]) ~ multi_normal(zeros_K, Phi_cov);
  for (j in 1:J) DD[, j] ~ bernoulli_logit(yy[, j]);
}
