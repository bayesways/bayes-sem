data {
  int<lower=1> N;
  int<lower=1> K;
  int<lower=1> J;
  int<lower=0, upper=1> DD[N, J];
}

transformed data{
  real<lower=0> c = square(0.2);
  vector[J] zeros = rep_vector(0, J);
  cov_matrix[J] I_c = diag_matrix(rep_vector(c, J));
}

parameters {
  vector[J] alpha;
  matrix[J,K] beta;
  cov_matrix[J] Sigma_u;
  matrix[N,K] zz;
  matrix[N,J] uu;
}

transformed parameters{
  matrix[N,J] yy;
  for (n in 1:N){
    yy[n,] = to_row_vector(alpha) + zz[n,] * beta' + uu[n,];
  }
}

model {
  to_vector(beta) ~ normal(0, 1);
  to_vector(zz) ~ normal(0, 1);
  to_vector(alpha) ~ normal(0, 1);
  Sigma_u ~ inv_wishart(J+6, I_c);
  for (n in 1:N){
    to_vector(uu[n,]) ~ multi_normal(zeros, Sigma_u);
    }
  // equivalent to_vector(uu) ~ normal(0, 0.2);

  for (j in 1:J) DD[, j] ~ bernoulli_logit(yy[, j]);
}

generated quantities{
  matrix[J,J] Omega;
  Omega = beta * beta';
}
