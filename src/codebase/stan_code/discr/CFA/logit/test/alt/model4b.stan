data {
  int<lower=1> N;
  int<lower=1> K;
  int<lower=1> J;
  int<lower=0, upper=1> DD[N, J];
  real<lower=0> c;
}

transformed data{
  cov_matrix[J] I_J = diag_matrix(rep_vector(1, J));
}

parameters {
  vector[J] alpha;
  matrix[3,K] beta_free; // 3 free eleements per factor
  matrix[J-3,K] beta_zeros; // 3 zero elements per factor
  matrix[N,J] yystar;
  cov_matrix[J] Omega_cov;
}

transformed parameters{
  matrix[J,K] beta;
  cholesky_factor_cov [J] L;
  matrix[N,J] yy;

  for(j in 1:J) {
    for (k in 1:K) beta[j,k] = 0;
  }
  // set the free elements
  for (k in 1:K) beta[1+3*(k-1) : 3+3*(k-1), k] = beta_free[1:3,k];
  // set the zero elements
  beta[4:J, 1] = beta_zeros[1:(J-3), 1];
  beta[1:(J-3), K] = beta_zeros[1:(J-3), K];
  
  L = cholesky_decompose(Omega_cov);

  for (n in 1:N) yy[n,] = to_row_vector(alpha) + yystar[n,] * L;

}

model {
  to_vector(beta_free) ~ normal(0, 1);
  to_vector(beta_zeros) ~ normal(0, 0.1);
  to_vector(alpha) ~ normal(0, 10);
  Omega_cov ~ inv_wishart(J+6, I_J);
  to_row_vector(yystar) ~ normal(0,1);
  for (j in 1:J) DD[, j] ~ bernoulli_logit(yy[, j]);
}

generated quantities{
  matrix[J,J] betabeta =  beta * beta';
  cov_matrix[J] Marg_cov = beta * beta'+ Omega_cov;
}

