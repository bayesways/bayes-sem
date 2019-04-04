data {
  int<lower=1> N;
  int<lower=1> J;
  matrix[N,J] yy;
}

transformed data{
  vector[J] zeros = rep_vector(0, J);
  matrix[J, J] I_c = diag_matrix(rep_vector(0.01, J));
}

parameters {
  vector<lower=0>[J] sigma;
  real<lower=0> sigma_z;
  vector[J] mu;
  vector[J] uu[N];
  cov_matrix[J] Sigma_u;
  vector<lower=0>[J-1] beta_t;
}

transformed parameters{
  vector<lower=0>[J] beta;
  cov_matrix[J] Sigma_epsilon;
  cov_matrix[J] Omega;
  Sigma_epsilon = diag_matrix(square(sigma));
  beta = append_row(1, beta_t);
  Omega = dot_product(beta, beta') * square(sigma_z) + Sigma_epsilon;
}

model {
  sigma ~ cauchy(0, 4);
  sigma_z ~ cauchy(0,4);
  to_row_vector(beta_t) ~ normal(0, 1);
  to_row_vector(mu) ~ normal(0, 1);
  Sigma_u ~ inv_wishart(J+6, I_c);
  for (n in 1:N){
  to_row_vector(uu[n,]) ~ multi_normal(zeros, Sigma_u);
  }
  for (n in 1:N){
    yy[n,] ~ multi_normal(mu+uu[n,], Omega);
  }
}
