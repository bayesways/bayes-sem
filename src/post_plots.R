library('ggplot2')
library('dplyr')
library('tidyr')
library('ggthemes')
library('scales')
library('reshape2')
library('readr')
library(forcats)

df <- read_csv("src/log/plot_data/post_df.csv") %>%
  mutate(error = data - mean, 
         index = factor(1:n()))

labels = paste0('item(' ,df$J,',', df$K, ')')

ggplot(df) +
  geom_segment(aes(x = `q2.5` , xend = `q97.5`, y=index, yend=index)) +
  geom_point(aes(y=index, x=mean), color = 'black', alpha=.5)+
  geom_point(aes(y=index, x=data), shape=24, fill ='#0394fc', color =  '#0394fc', alpha=.5)+
  scale_y_discrete(labels = labels)+
  theme_fivethirtyeight()+
  scale_colour_manual(values = c('black', '#0394fc'),
                      labels = c("data", "data2"))
  
ggsave("doc/sim1.png", width = 8, height = 12, units = "cm")

