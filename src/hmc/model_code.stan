data {
  int<lower=1> N;
  int<lower=1> K;
  int<lower=1> J;
  matrix[N,J] yy;

}

parameters {
  matrix[J,K] beta;
  vector<lower=0>[J] sigma;
}

transformed parameters{
  matrix[J,J] Omega;
  
  Omega = beta * beta' + diag_matrix(sigma .* sigma);
}

model {
  to_row_vector(beta) ~ normal(0, 1);
  sigma ~ cauchy(0, 10);
  for (n in 1:N){
    yy[n,] ~ multi_normal(rep_vector(0, J), Omega);
  }
}
