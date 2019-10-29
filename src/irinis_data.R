library('dplyr')
library('readr')
library("openxlsx")
form1.df <- openxlsx::read.xlsx("dat/Forms.xlsx", sheet = "form1")

Y.r <- as.matrix(form1.df[,211:380])
xi.true <- form1.df$Flagged

time.r <- form1.df[411:580] #' 1636x170

n.ZERO <- sum(rowSums(time.r==0)!=0)      # 12
id.ZERO <- (1:nrow(Y.r))[rowSums(time.r==0)!=0] # 5,11,61,64,142,149,182,219,269,276,292,1310

#' Delete ppl of zero response time
Y.real <- Y.r[-id.ZERO,] # 1624 x 170
t.real <- time.r[-id.ZERO,]
xi.real <- xi.true[-id.ZERO]

df<-as_tibble(cbind(flagged = xi.real, Y.real))
names(df) <- c('flag', paste0('item_', seq(1:170)))

df <- df %>% 
  mutate(subj_id = 1:n()) %>%
  select(subj_id, everything())

write_csv(df, 'dat/irinis_test_data.csv')
