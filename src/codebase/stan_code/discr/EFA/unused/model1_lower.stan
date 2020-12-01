data {
  int<lower=1> N;
  int<lower=1> K;
  int<lower=1> J;
  int<lower=0, upper=1> DD[N, J];
}

parameters {
  vector[J] alpha;
  matrix[N,K] zz;
  vector<lower=0>[2] beta_leading;
  vector[J-1] beta_free1; // 1st factor
  vector[J-2] beta_free2; // 2nd factor (enforce beta lower triangular)
}

transformed parameters{
  matrix[N,J] yy;
  matrix[J,K] beta;
  beta[1,1] = beta_leading[1];
  beta[2:J, 1] = beta_free1;
  beta[1,2] = 0;
  beta[2,2] = beta_leading[2];
  beta[3:J, 2] = beta_free2;
  for (n in 1:N) yy[n,] = to_row_vector(alpha) + zz[n,] * beta';
}
  
model {
  to_vector(beta_leading) ~ normal(0, 2);
  to_vector(beta_free1) ~ normal(0, 1);
  to_vector(beta_free2) ~ normal(0, 1);
  to_vector(alpha) ~ normal(0, 10);
  to_vector(zz) ~ normal(0, 1);
  for (j in 1:J) DD[, j] ~ bernoulli_logit(yy[, j]);
}
