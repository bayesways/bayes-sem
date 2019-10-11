data {
  int<lower=1> N;
  int<lower=1> K;
  int<lower=1> J;
  matrix[N,J] yy;
}

transformed data{
  cov_matrix[J] I = diag_matrix(rep_vector(1, J));
}

parameters {
  cov_matrix[J] Sigma;
  vector[J] alpha;
}

model {
  to_vector(alpha) ~ normal(0, 10);
  Sigma ~ inv_wishart(J+2, I);
  for (n in 1:N){
    yy[n, ] ~ multi_normal(alpha,  Sigma);
  }
}
