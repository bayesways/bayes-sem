data {
  int<lower=1> N;
  int<lower=1> K;
  int<lower=1> J;
  matrix[N,J] yy;

}

transformed data {
  vector[J] sigma;
  matrix[J, J] Sigma = diag_matrix(sigma);
}


parameters {
  vector[J] mu;
  vector[J] uu[N];
  matrix[J,K] beta;
}

transformed parameters{
  matrix[J,J] Omega;
  Omega = beta * beta' + Sigma;
}

model {
  to_row_vector(beta) ~ normal(0, 1);
  to_row_vector(mu) ~ normal(0, 100);
  for (n in 1:N){
  to_row_vector(uu[n,]) ~ normal(0, 0.2);
  }
  for (n in 1:N){
    yy[n,] ~ multi_normal(mu+uu[n,], Omega);
  }
}
