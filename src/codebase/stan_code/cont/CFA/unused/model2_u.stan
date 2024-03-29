data {
  int<lower=1> N;
  int<lower=1> K;
  int<lower=1> J;
  matrix[N,J] yy;
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
  matrix[N,J] uu;
  cov_matrix[J] Omega;
}

transformed parameters{
  cov_matrix[J] Theta;
  matrix[J,K] beta;
  cov_matrix[J] Marg_cov;
  
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
  to_vector(alpha) ~ normal(0, 10);
  for(j in 1:J) sigma_square[j] ~ inv_gamma(c0, (c0-1)/sigma_prior[j]);
  Phi_cov ~ inv_wishart(J+4, I_K);
  Omega ~ inv_wishart(J+6, I_J);
  for (n in 1:N){
    to_vector(uu[n,]) ~ multi_normal(zeros, Omega);
  }
  for (n in 1:N){
    yy[n, ] ~ multi_normal(alpha + to_vector(uu[n,]),  Marg_cov);
  }
  
}

generated quantities{
  vector<lower=0>[J] sigma = sqrt(sigma_square);
}

