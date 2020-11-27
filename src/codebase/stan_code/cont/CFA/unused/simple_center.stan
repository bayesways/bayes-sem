data {
  int<lower=1> N;
  int<lower=1> K;
  matrix[N,K] yy;
  vector[K] sigma_prior;

}

transformed data{
  cov_matrix[K] I = diag_matrix(rep_vector(1, K));
  vector[K] zeros = rep_vector(0, K);
  real<lower=0> c0 = 2.5;
}

parameters {
  // cholesky_factor_corr[K] Phi_corr_chol;
  // vector<lower=0>[K] sigma_square;
  vector<lower=0>[K] sigma;
  vector[K] alpha;
}

transformed parameters{
  matrix[N,K] yy_centered ;
  for (n in 1:N){
    for (k in 1:K){
      // yy_centered[n,k] =  (yy[n,k] - alpha[k]) / sqrt(sigma_square[k]);  
      yy_centered[n,k] =  (yy[n,k] - alpha[k]) / sigma[k];
    }
  }
}

model {
  alpha ~ normal(0, 10);
  // for(k in 1:K) sigma_square[k] ~ inv_gamma(c0, (c0-1)/sigma_prior[k]);
  sigma ~ cauchy(0,.1);
  for (n in 1:N){
    // yy_centered[n, ] ~ multi_normal_cholesky(zeros, I);
    yy_centered[n, ] ~ multi_normal(zeros, I);

  }
}

// generated quantities{
//   matrix [K, K] Phi_cov = multiply_lower_tri_self_transpose(diag_pre_multiply(sigma, Phi_corr_chol));
//   
// }
