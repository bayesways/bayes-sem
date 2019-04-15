data {
  int<lower=1> N;
  int<lower=1> K;
  int<lower=1> J;
  matrix[N,J] yy;
}

transformed data{
  real<lower=0> c = 0.01;
  vector[J] zeros = rep_vector(0, J);
  cov_matrix[J] I_c = diag_matrix(rep_vector(c, J));
}

parameters {
  vector<lower=0>[J] sigma;
  vector[J] alpha;
  vector[J] beta1;
  vector[J-3] beta2;
}

transformed parameters{
  matrix[J,K] beta;
  cov_matrix[J] Sigma_epsilon = diag_matrix(square(sigma));
  cov_matrix[J] Omega; 
  for (j in 1:J){
    for (k in 1:K) beta[j,k] = 0;
  }
  beta[1:J,1] = beta1;
  beta[4:J,2] = beta2;
  
  Omega = beta * beta'+ Sigma_epsilon;
}

model {
  to_row_vector(beta1) ~ normal(0, 1);
  to_row_vector(beta2) ~ normal(0, 1);
  to_row_vector(alpha) ~ normal(0, 1);
  for (n in 1:N){
    yy[n, ] ~ multi_normal(alpha,  Omega);
  }
}

generated quantities{
  matrix [J, J] Omega_beta = beta * beta';
}
