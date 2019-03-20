data {
  int<lower=1> N;
  int<lower=1> K;
  int<lower=1> J;
  int<lower=0, upper=1> DD[N, J];

}

transformed data {
  matrix[J, J] Sigma = diag_matrix(rep_vector(0.04, J));
}


parameters {
  vector[J] mu;
  vector[J] uu[N];
  matrix[J,K] beta;
  matrix[N,J] yy;
}

transformed parameters{
  matrix[J,J] Omega;
  Omega = beta * beta' + Sigma;
}

model {
  to_row_vector(beta) ~ normal(0, 1);
  to_row_vector(mu) ~ normal(0, 100);
  for (n in 1:N){
  to_row_vector(uu[n,]) ~ normal(0, 0.02);
  }
  for (n in 1:N){
    yy[n,] ~ multi_normal(mu+uu[n,], Omega);
  }
  for (j in 1:J) DD[, j] ~ bernoulli_logit(yy[, j]);
}
