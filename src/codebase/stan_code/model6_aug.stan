data {
  int<lower=1> N;
  int<lower=1> K;
  int<lower=1> J;
  int<lower=0, upper=1> DD[N, J];

}

transformed data {
  matrix[J, J] I_c = diag_matrix(rep_vector(0.04, J));
}

parameters {
  vector[J] mu;
  matrix[J,K] beta;
  matrix[N,K] zz;
  matrix[N,J] uu;

}

transformed parameters{
  matrix[N,J] yy;
  for (n in 1:N){
    yy[n,] = to_row_vector(mu) + zz[n,] * beta' + uu[n,];
  }
}

model {
  to_row_vector(beta) ~ normal(0, 1);
  to_row_vector(zz) ~ normal(0, 1);
  to_row_vector(mu) ~ normal(0, 100);
  to_row_vector(uu) ~ normal(0, 0.2);

  for (j in 1:J) DD[, j] ~ bernoulli_logit(yy[, j]);
}

generated quantities{
  matrix[J,J] Omega;
  Omega = beta * beta';
}
