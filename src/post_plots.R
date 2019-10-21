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

##############################
##############################

df <- read_csv("src/log/plot_data/post_df_sim2.csv") %>%
  mutate(error = data - mean, 
         index = factor(1:n()))

df
pr_zero <-as.numeric(quantile(rnorm(10000, mean = 0, sd = 0.1),
                         probs = c(.025, 0.975)))
pr_free <-as.numeric(quantile(rnorm(10000),
                         probs = c(.025, 0.975)))
df<- df %>%
  mutate(prior1 =  case_when(
  index %in% c('2', '3', '11', '12') ~ pr_free[1],
  index %in% c('1','10') ~ 1,
    TRUE ~ pr_zero[1]),
  prior2 =  case_when(
    index %in% c('2', '3', '11', '12') ~ pr_free[2],
    index %in% c('1','10') ~ 1,
    TRUE ~ pr_zero[2]),
  )

labels = paste0('item(' ,df$J,',', df$K, ')')

ggplot(df) +
  geom_segment(aes(x = `q2.5` , xend = `q97.5`, y=index, yend=index), size = 1.5, alpha = 0.6) +
  geom_segment(aes(x = prior1 , xend = prior2, y=index, yend=index), color = 'black', size = 1, alpha = 0.5) +
  geom_point(aes(y=index, x=prior1), shape=3, color = 'black', alpha=.5)+
  geom_point(aes(y=index, x=prior2), shape=3, color = 'black', alpha=.5)+
  geom_point(aes(y=index, x=mean), color = 'black', alpha=.5, size = 2.5)+
  geom_point(aes(y=index, x=data), shape=24, fill ='#0394fc', color =  '#0394fc', alpha=.5, size = 2.5)+
  scale_y_discrete(labels = labels)+
  theme_fivethirtyeight()+
  scale_colour_manual(values = c('black', '#0394fc'),
                      labels = c("data", "data2"))
  

ggsave("doc/sim2.png", width = 8, height = 12, units = "cm")


post_color <- '#020f4d'
data_color <-'#c28f30'

ggplot(df) +
  theme_economist_white(gray_bg = F)+
  geom_segment(aes(x = `q2.5` , xend = `q97.5`, y=index, yend=index),
               color = post_color, size = 1.5, alpha = 0.6) +
  geom_segment(aes(x = prior1 , xend = prior2, y=index, yend=index),
               color = post_color, size = 1, alpha = 0.5) +
  geom_point(aes(y=index, x=prior1), shape=3, color = post_color, alpha=.5)+
  geom_point(aes(y=index, x=prior2), shape=3, color = post_color, alpha=.5)+
  geom_point(aes(y=index, x=mean), color = post_color, alpha=.5, size = 2.5)+
  geom_point(aes(y=index, x=data), shape=24, fill =data_color, color =  data_color, alpha=.5, size = 2.5)+
  scale_y_discrete(labels = labels)+
  scale_x_continuous(breaks = pretty_breaks(6),
                   expand = c(.1,0.1))+
  scale_colour_manual(values = c(post_color, data_color),
                      labels = c("data", "data2"))+
  theme(legend.position = 'none',
          axis.text = element_text(size=12),
          axis.title = element_blank(),
          axis.ticks.length = unit( 10 * 0.5, "points"),
          panel.grid = element_blank(),
          # panel.grid.major.x = element_blank(),
          panel.grid.major.x = element_line(colour = "grey85", size = rel(.5)),
          panel.grid.major.y = element_line(colour = "grey85", size = rel(.5)),
          axis.text.y = element_text(hjust = .8,
                                     color= 'grey40'),
          axis.text.x = element_text(hjust = 0.5,
                                     color= 'grey40'),
          plot.title = element_text(family = "Verdana")) 


ggsave("doc/sim2_theme2.png", width = 10, height = 12, units = "cm")

