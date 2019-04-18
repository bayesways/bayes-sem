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
  matrix[K,K] beta_t;
  matrix[N,J] uu;

}

transformed parameters{
  matrix[J,K] beta;
  cov_matrix[J] Sigma_epsilon = diag_matrix(square(sigma));
  cov_matrix[J] Omega = beta * beta'+ Sigma_epsilon;
  for (j in 1:J){
    for (k in 1:K) beta[j,k] = 0;
  }
  beta[1,1] = 1;
  beta[2:3,1] = beta_t[1:2,1];
  beta[4,2] = 1;
  beta[5:6,2] = beta_t[1:2,2];
}

model {
  to_row_vector(beta_t) ~ normal(0, 1);
  to_row_vector(alpha) ~ normal(0, 1);
  for (n in 1:N){
    to_vector(uu[n,]) ~ multi_normal(zeros, I_c);
  }
  for (n in 1:N){
    yy[n, ] ~ multi_normal(alpha + to_vector(uu[n,]),  Omega);
  }
}

generated quantities{
  matrix [J, J] Omega_beta = beta * beta';
}
