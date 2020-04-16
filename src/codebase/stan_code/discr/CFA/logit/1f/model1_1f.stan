data {
  int<lower=1> N;
  int<lower=1> K;
  int<lower=1> J;
  int<lower=0, upper=1> DD[N, J];
}

parameters {
  vector[J] alpha;
  vector[J-1] beta_free;
  vector[N] zz;
}

transformed parameters{
  matrix[N,J] yy;
  vector[J] beta;
  // set ones
  beta[1] = 1;
  // set the free elements
  beta[2 : J] = beta_free[1:(J-1)];
  for (n in 1:N) yy[n,] = to_row_vector(alpha) + to_row_vector(zz[n]*beta);
}

model {
  to_vector(beta_free) ~ normal(0, 1);
  to_vector(alpha) ~ normal(0, 10);
  to_vector(zz) ~ normal(0, 1);
  for (j in 1:J) DD[, j] ~ bernoulli_logit(yy[, j]);
}
