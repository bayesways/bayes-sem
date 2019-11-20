data {
  int<lower=1> N;
  int<lower=1> K;
  int<lower=1> J;
  int<lower=0, upper=1> DD[N, J];
}

transformed data{
  vector[J] zeros_J = rep_vector(0, J);
  cov_matrix[J] I_J = diag_matrix(rep_vector(1, J));
}

parameters {
  vector[J] alpha;
  matrix[J,K] beta;
  cov_matrix[J] Omega;
  matrix[N,K] zz;
  matrix[N,J] uu;
}

transformed parameters{
  matrix[N,J] yy;
  for (n in 1:N) yy[n,] = to_row_vector(alpha) + zz[n,] * beta' + uu[n,];
}

model {
  to_vector(beta) ~ normal(0, 1);
  to_vector(alpha) ~ normal(0, 10);
  to_vector(zz) ~ normal(0,1);
  Omega ~ inv_wishart(J+6, I_J);
  for (n in 1:N) to_vector(uu[n,]) ~ multi_normal(zeros_J, Omega);
  for (j in 1:J) DD[, j] ~ bernoulli_logit(yy[, j]);
}
