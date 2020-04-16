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
  vector[J-1] beta_free;
  vector[N] zz;
  matrix[N,J] uu;
  cov_matrix[J] Omega_cov;
}

transformed parameters{
  matrix[N,J] yy;
  vector[J] beta;
  // set ones
  beta[1] = 1;
  // set the free elements
  beta[2 : J] = beta_free[1:(J-1)];
  for (n in 1:N) yy[n,] = to_row_vector(alpha) + to_row_vector(zz[n]*beta) + uu[n,];
}

model {
  to_vector(beta_free) ~ normal(0, 1);
  to_vector(alpha) ~ normal(0, 10);
  to_vector(zz) ~ normal(0, 1);
  Omega_cov ~ inv_wishart(J+6, I_J);
  for (n in 1:N) uu[n,] ~ multi_normal(zeros_J, Omega_cov);
  for (j in 1:J) DD[, j] ~ bernoulli_logit(yy[, j]);
}
