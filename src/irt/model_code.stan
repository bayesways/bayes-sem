data {
  int<lower=1> N;
  int<lower=1> K;
  int<lower=1> J;
  int<lower=0, upper=1> DD[N, J];

}

transformed data {
  matrix[J, J] Sigma = diag_matrix(rep_vector(1, J));
}


parameters {
  matrix[J,K] beta;
  matrix[N,J] yy;
}

transformed parameters{
  matrix[J,J] Omega;
  Omega = beta * beta' + Sigma;
}

model {
  to_row_vector(beta) ~ normal(0, 1);
  for (n in 1:N){
    yy[n,] ~ multi_normal(rep_vector(0, J), Omega);
  }
  for (j in 1:J) DD[, j] ~ bernoulli_logit(yy[, j]);
}
