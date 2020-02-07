data {
  int<lower=1> N;
  int<lower=1> K;
  int<lower=1> J;
  int<lower=0, upper=1> DD[N, J];
}

parameters {
  vector[J] alpha;
  matrix[3,K] beta_free; // 3 free eleements per factor
  matrix[N,K] zz;
}

transformed parameters{
  matrix[J,K] beta;
  matrix[N,J] yy;

  for(j in 1:J) {
    for (k in 1:K) beta[j,k] = 0;
  }
  // set the free elements
  for (k in 1:K) beta[1+3*(k-1) : 3+3*(k-1), k] = beta_free[1:3,k];

  for (n in 1:N) yy[n,] = to_row_vector(alpha) + zz[n,] * beta';
}
  
model {
  to_vector(beta_free) ~ normal(0, 1);
  to_vector(alpha) ~ normal(0, 10);
  to_vector(zz) ~ normal(0, 1);
  for (j in 1:J) DD[, j] ~ bernoulli_logit(yy[, j]);
  
}

