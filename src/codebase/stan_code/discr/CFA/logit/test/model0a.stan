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
  matrix[N,J] yystar;
}

transformed parameters{
  matrix[N,J] yy;
  for (n in 1:N) yy[n,] = to_row_vector(alpha) + yystar[n,] * L;

}

model {
  alpha ~ normal(0,10);
  L ~ lkj_corr_cholesky(2);
  to_row_vector(yystar) ~ normal(0,1);
  for (j in 1:J) DD[, j] ~ bernoulli_logit(yy[, j]);
}

generated quantities{
  corr_matrix[J] R = multiply_lower_tri_self_transpose(L);
}
