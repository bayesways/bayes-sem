data {
  int<lower=1> N;
  int<lower=1> K;
  int<lower=1> J;
  int<lower=0, upper=1> DD[N, J];
}

transformed data{
  vector[J] zeros_J = rep_vector(0, J);
  vector[K] zeros_K = rep_vector(0, K);
  cov_matrix[J] I_J = diag_matrix(rep_vector(1, J));
  cov_matrix[K] I_K = diag_matrix(rep_vector(1, K));
}

parameters {
  vector[J] alpha;
  matrix[J,K] beta;
  cov_matrix[J] Omega;
  matrix[N,J] yy;
}

transformed parameters{
  cov_matrix[J] Marg_cov;
  Marg_cov = beta * beta' + Omega;
}

model {
  to_vector(beta) ~ normal(0, 1);
  to_vector(alpha) ~ normal(0, 10);
  Omega ~ inv_wishart(J+6, I_J);
  for (n in 1:N) yy[n, ] ~ multi_normal(alpha,  Marg_cov);
  for (j in 1:J) DD[, j] ~ bernoulli_logit(yy[, j]);
}
