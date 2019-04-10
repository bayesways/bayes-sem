data {
  int<lower=1> N;
  int<lower=1> K;
  int<lower=1> J;
  matrix[N,J] yy;
}

transformed data{
  real<lower=0> c = 0.01;
  vector[J] zeros = rep_vector(0, J);
  cov_matrix[J] I_c = diag_matrix(rep_vector(c, J));
}

parameters {
  vector<lower=0>[J] sigma;
  vector[J] alpha;
  matrix[J,K] beta;
  matrix[N,J] uu;
  matrix[N,K] zz;

}

transformed parameters{
  cov_matrix[J] Sigma_epsilon = diag_matrix(square(sigma));
  matrix[N,J] mean_yy;
  for (n in 1:N){
    mean_yy[n,] = to_row_vector(alpha) + zz[n,] * beta' + uu[n,];
  }
}

model {
  to_vector(beta) ~ normal(0, 1);
  to_vector(alpha) ~ normal(0, 1);
  to_vector(zz) ~ normal(0, 1);
  for (n in 1:N){
    to_vector(uu[n,]) ~ multi_normal(zeros, I_c);
  }
  for (n in 1:N){
    yy[n, ] ~ multi_normal(mean_yy[n,],  Sigma_epsilon);
  }
}

generated quantities{
  matrix [J, J] Omega_beta = beta * beta';
}
