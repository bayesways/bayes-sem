library('ggplot2')
library('dplyr')
library('tidyr')
library('ggthemes')
library('scales')
library('reshape2')
library('readr')
library(forcats)

setwd("~/myPhD/bayes-sem/src/log/")

df <- read_csv("~/myPhD/bayes-sem/dat/heatmap_data.csv") 
colnames <- as.character(paste0('', seq(1:15)))
names(df) <- colnames


plot_data<-tibble::rowid_to_column(df, "ID") %>%
  gather(var, val, -ID) %>%
  mutate(var = fct_relevel( var, colnames))


last(plot_data$ID)

  
ggplot(plot_data, aes(var, ID)) +
  geom_tile(aes(fill = val), colour = "white") +
  scale_fill_gradient2()+
  theme_minimal()+ # minimal theme
  scale_x_discrete(breaks = pretty_breaks(10),
                   expand = c(0,0),)+
  scale_y_reverse()+
  theme(axis.text.x = element_text(angle = 90, vjust = 1,
                                   size = 16, hjust = 1),
        axis.text.y = element_text(size = 16))+
  theme(
    axis.title.x = element_blank(),
    axis.title.y = element_blank(),
    panel.grid.major = element_blank(),
    panel.border = element_blank(),
    panel.background = element_blank(),
    axis.ticks = element_blank(),
    legend.justification = c(1, 0),
    legend.position = 'none', 
    legend.direction = "vertical",
    plot.title = element_text(family = "Verdana", size = 16),
    legend.title=element_text(size=10))


########################################
########################################

ggplot(plot_data, aes(var, ID)) +
  geom_tile(aes(fill = abs(val)), colour = "white") +
  scale_fill_gradient(low = '#120a0b', high = '#f5425a' )+
  # scale_fill_gradient(low = "#ffffff", high = "#e02f38")+
  
  guides(fill = guide_colourbar(barwidth = 0.5, barheight = 10))+
  # scale_fill_viridis(option = 'D')+
  # scale_fill_jco()+
  # scale_fill_gradientn(colours = pal) + 
  theme_economist_white(gray_bg = F, horizontal = F) +
  scale_y_discrete(breaks = pretty_breaks(10),
                   expand = c(0,0),
                   reverse_trans())+
  theme(axis.text = element_text(size=10),
        legend.position = 'right',
        legend.title=element_blank(),
        legend.spacing.x = unit(.1, 'cm'),
        legend.key.size = unit(1, "cm"),
        legend.text = element_text(size = 10),
        axis.title = element_blank(),
        axis.ticks.length = unit( 10 * 0.5, "points"),
        panel.grid = element_blank(),
        panel.grid.major.x = element_blank(),
        panel.grid.major.y = element_line(),
        axis.text.y = element_text(hjust = .8,
                                   color= 'grey40'),
        axis.text.x = element_text(hjust = 0.5,
                                   color= 'grey40'),
        plot.title = element_text(size = 10),
        plot.subtitle = element_text(size = 10)
  )

########################################
########################################

ggplot(plot_data , aes(var, ID)) +
  geom_tile(aes(fill = val), colour = "white") +
  scale_fill_gradient2()+
  guides(fill = guide_colourbar(barwidth = 0.5, barheight = 10))+
  theme_economist_white(gray_bg = F, horizontal = F) +
  scale_y_discrete(breaks = pretty_breaks(10),
                   expand = c(0,0),
                   reverse_trans())+
theme(axis.text = element_text(size=6),
      legend.position = 'right',
      legend.title=element_blank(),
      legend.spacing.x = unit(.1, 'cm'),
      legend.key.size = unit(1, "cm"),
      legend.text = element_text(size = 6),
      axis.title = element_blank(),
      axis.ticks.length = unit( 10 * 0.5, "points"),
      panel.grid = element_blank(),
      panel.grid.major.x = element_blank(),
      panel.grid.major.y = element_line(),
      axis.text.y = element_text(hjust = .8,
                                 color= 'grey40'),
      axis.text.x = element_text(hjust = 0.5,
                                 color= 'grey40'),
      plot.title = element_text(size = 6),
      plot.subtitle = element_text(size = 6)
)

ggsave("act1.pdf", width = 8, height = 12, units = "cm")
ggsave("act1.png", width = 8, height = 12, units = "cm")



