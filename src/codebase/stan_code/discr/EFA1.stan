data {
  int<lower=1> N;
  int<lower=1> K;
  int<lower=1> J;
  int<lower=0, upper=1> DD[N, J];
}

transformed data{
  real<lower=0> c = 0.01;
  vector[J] zeros = rep_vector(0, J);
  matrix[J, J] I_c = diag_matrix(rep_vector(c, J));
}

parameters {
  vector<lower=0>[J] sigma;
  vector[J] mu;
  vector[J] uu[N];
  cov_matrix[J] Sigma_u;
  matrix[J-1,K] beta_t;
  matrix[N,J] yy;
}

transformed parameters{
  matrix[J,K] beta;
  matrix[J, J] Sigma_epsilon = diag_matrix(sigma .* sigma);
  matrix[J, J] Omega = beta * beta' + Sigma_epsilon;
  for (j in 1:J){
  beta[,j] = append_row(1, beta_t[, j]);
  }
  
}

model {
  sigma ~ cauchy(0, 10);
  to_row_vector(beta_t) ~ normal(0, 1);
  to_row_vector(mu) ~ normal(0, 100);
  Sigma_u ~ inv_wishart(J+6, I_c);
  for (n in 1:N){
  to_row_vector(uu[n,]) ~ multi_normal(zeros, Sigma_u);
  }
  for (n in 1:N){
    yy[n,] ~ multi_normal(mu+uu[n,], Omega);
  }
}
