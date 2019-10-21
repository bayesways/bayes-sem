data {
  int<lower=1> N;
  int<lower=1> K;
  int<lower=1> J;
  int<lower=0, upper=1> DD[N, J];
  vector[J] sigma_prior;

}

transformed data{
  vector[J] zeros = rep_vector(0, J);
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
  cov_matrix[J] Omega;
  matrix[N,K] zz;
  matrix[N,J] uu;
}

transformed parameters{
  cov_matrix[J] Theta;
  matrix[J,K] beta;
  matrix[N,J] yy;

  
  Theta = diag_matrix(sigma_square);

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

  for (n in 1:N){
    yy[n,] = to_row_vector(alpha) + zz[n,] * beta' + uu[n,];
  }
}

model {
  to_vector(beta_free) ~ normal(0, 1);
  to_vector(beta_zeros) ~ normal(0, 0.1);
  to_vector(alpha) ~ normal(0, 10);
  for(j in 1:J) sigma_square[j] ~ inv_gamma(c0, (c0-1)/sigma_prior[j]);
  Phi_cov ~ inv_wishart(J+4, I_K);
  Omega ~ inv_wishart(J+6, I_J);
  for (n in 1:N){
    to_vector(uu[n,]) ~ multi_normal(zeros, Omega);
  }
  for (j in 1:J) DD[, j] ~ bernoulli_logit(yy[, j]);
}

generated quantities{
  vector<lower=0>[J] sigma = sqrt(sigma_square);
}


