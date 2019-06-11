data {
  int<lower=1> N;
  int<lower=1> M;
  int<lower=1> K;
  int<lower=1> J;
  int<lower=1, upper=M> DD[N, J];
}

transformed data{
  cov_matrix[J] I_c = diag_matrix(rep_vector(1, J));
}

parameters {
  vector[J] alpha;
  matrix[J,K] beta;
  matrix[N,J] yy;
  ordered[M-1] cutpoints;
}

transformed parameters{
  cov_matrix[J] Omega = beta * beta'+ I_c;
}

model {
  to_row_vector(beta) ~ normal(0, 1);
  to_row_vector(alpha) ~ normal(0, 1);
  for (n in 1:N) yy[n, ] ~ multi_normal(alpha, Omega);
  for (j in 1:J) DD[, j] ~ ordered_logistic(yy[, j], cutpoints);
}

generated quantities{
  matrix [J, J] Omega_beta = beta * beta';
}

