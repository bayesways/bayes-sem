data {
  int<lower=1> N;
  int<lower=1> K;
  int<lower=1> J;
  matrix[N,J] yy;
}

transformed data{
  cov_matrix[J] I = diag_matrix(rep_vector(1, J));
  vector[J] zeros = rep_vector(0, J);
}

parameters {
  cov_matrix[J] Sigma;
}

model {
  Sigma ~ inv_wishart(J+2, I);
  for (n in 1:N){
    yy[n, ] ~ multi_normal(zeros,  Sigma);
  }
}
