#'################################
#'
#' Data set 1
#'
#'################################
library("openxlsx")
form1.df <- openxlsx::read.xlsx("dat/Forms.xlsx", sheet = "form1")

item_form1.df <- openxlsx::read.xlsx("dat/Item_Info.xlsx", sheet = "Form1")

Y.r <- as.matrix(form1.df[,211:380])
xi.true <- form1.df$Flagged
eta.true <- item_form1.df$Flagged[1:170]

time.r <- form1.df[411:580] #' 1636x170

n.ZERO <- sum(rowSums(time.r==0)!=0)      # 12
id.ZERO <- (1:nrow(Y.r))[rowSums(time.r==0)!=0] # 5,11,61,64,142,149,182,219,269,276,292,1310

#' Delete ppl of zero response time
Y.real <- Y.r[-id.ZERO,] # 1624 x 170
t.real <- time.r[-id.ZERO,]
xi.real <- xi.true[-id.ZERO]
eta.real <- eta.true

Y.total <- rowSums(Y.real);
par(mfrow = c(1, 2))
hist(Y.total, main = "(a)"),


Cheater = xi.real;
Cheater[xi.real==0] = "N"
Cheater[xi.real==1] = "Y"
  
Y = data.frame(total = Y.total, Cheater = Cheater);

pdf(file = "histpeople.pdf", height = 4, width = 5)
p = ggplot(Y, xlab = "Total Score", aes(x = total)) +
  geom_histogram(aes(color = Cheater, fill = Cheater), 
                 position = "identity", bins = 30, alpha = 0.4) +
  scale_color_manual(values = c("#00AFBB", "#E7B800")) +
  scale_fill_manual(values = c("#00AFBB", "#E7B800"))
p + labs(x = "Total Score")
dev.off()


I.total <- colSums(Y.real)/nrow(Y.real);

Compromised = eta.real;
Compromised[eta.real==0] = "N"
Compromised[eta.real==1] = "Y"




Ptime <- rowMeans(log(t.real));
Y = data.frame(total = Ptime, Cheater = Cheater);

pdf(file = "histpeople2.pdf", height = 4, width = 5)
p = ggplot(Y, aes(x = total)) +
  geom_histogram(aes(color = Cheater, fill = Cheater), 
                 position = "identity", bins = 30, alpha = 0.4) +
  scale_color_manual(values = c("#00AFBB", "#E7B800")) +
  scale_fill_manual(values = c("#00AFBB", "#E7B800"))
p + labs(x = "Mean Log Response Time")
dev.off()

I.total <- colMeans(log(t.real));
Item = data.frame(total = I.total, Compromised = Compromised);

pdf(file = "histitem2.pdf", height = 4, width = 5)
p = ggplot(Item, aes(x = total)) +
  geom_histogram(aes(color = Compromised, fill = Compromised), 
                 position = "identity", bins = 15, alpha = 0.4) +
  scale_color_manual(values = c("#00AFBB", "#E7B800")) +
  scale_fill_manual(values = c("#00AFBB", "#E7B800"))
p + labs(x = "Mean Log Response Time")
dev.off()

roc(factor(Cheater), rowMeans(Y.real))$auc
roc(factor(Cheater), rowMeans(log(t.real)))$auc

roc(factor(Compromised), colMeans(Y.real))$auc
roc(factor(Compromised), colMeans(log(t.real)))$auc


#Fitting a Rasch model 
install.packages("eRm");
library(eRm)

Raschres <- RM(Y.real);

betas <- -coef(Raschres);
thetas <- person.parameter(Raschres)$thetapar[[1]];
thetas = thetas - mean(thetas);
betas = betas+mean(thetas);

#We will now look at the residuals to get an initial value. 
#We only look at the positive residuals. 

N = length(thetas);
J = length(betas);

temp = matrix(thetas, N, 1) %*% t(rep(1, J)) - rep(1, N) %*%  matrix(betas, 1, J)
prob.matr <- 1/(1+ exp(-temp));


#Look at residuals. 

quan <- quantile(betas, probs = c(0.33, 0.67));
easy.data = Y.real[,betas <= quan[1]]
diff.data = Y.real[,betas >= quan[2]]


plot(rowMeans(easy.data), rowMeans(diff.data))
points(rowMeans(easy.data)[xi.real==1], rowMeans(diff.data)[xi.real==1], col = "red")

peo.res = rowMeans(easy.data) - rowMeans(diff.data)
boxplot(peo.res~xi.real)

resid = -Y.real * log(prob.matr) - (1-Y.real) * log(1-prob.matr);
var.vec = rep(0, N);
for(i in 1:N){
  var.vec[i] = var(resid[i,])
}
boxplot(var.vec~xi.real)

  

index <- whi


item.res = colSums(resid>1.5);
boxplot(item.res~eta.real)




item.res = colSums(resid);
boxplot(item.res~eta.real)



















