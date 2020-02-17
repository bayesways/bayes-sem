data {
  int<lower=1> N;
  int<lower=1> J;
  int<lower=0, upper=1> DD[N, J];

}

transformed data{
  vector[J] zeros_J = rep_vector(0, J);
}

parameters {
  vector[J] alpha;
  cholesky_factor_corr[J] L;
  matrix[N,J] yy;
}

// transformed parameters{
// }

model {
  alpha ~ normal(0,10);
  L ~ lkj_corr_cholesky(2);
  for (n in 1:N) yy[n,] ~ multi_normal_cholesky(alpha, L);
  for (j in 1:J) DD[, j] ~ bernoulli_logit(yy[, j]);
}

generated quantities{
  corr_matrix[J] R = multiply_lower_tri_self_transpose(L);
}
