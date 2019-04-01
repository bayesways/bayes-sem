data {
  int<lower=1> N;
  int<lower=1> K;
  int<lower=1> J;
  matrix[N,J] yy;
}

transformed data{
  vector[J] zeros = rep_vector(0, J);
  matrix[J, J] I_c = diag_matrix(rep_vector(0.1, J));
}

parameters {
  vector<lower=0>[J] sigma;
  vector[J] mu;
  vector[J] uu[N];
  cov_matrix[J] Sigma_u;
  matrix[J,K] beta;
}

transformed parameters{
  matrix[J, J] Sigma_epsilon = diag_matrix(sigma .* sigma);
  matrix[J, J] Omega = beta * beta' + Sigma_epsilon;
}

model {
  sigma ~ cauchy(0, 10);
  to_row_vector(beta) ~ normal(0, 1);
  to_row_vector(mu) ~ normal(0, 100);
  Sigma_u ~ inv_wishart(J+6, I_c);
  for (n in 1:N){
  to_row_vector(uu[n,]) ~ multi_normal(zeros, Sigma_u);
  }
  for (n in 1:N){
    yy[n,] ~ multi_normal(mu+uu[n,], Omega);
  }
}
