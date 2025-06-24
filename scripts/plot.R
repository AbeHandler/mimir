library(ggplot2)

# Read CSV file into a data frame
df <- read.csv("copywritetraps.csv")

# Plot
p <- ggplot(df, aes(x = x, y = y)) +
  geom_line(color = "grey", size = 1.2) +
  geom_point(size = 3, color = "black") +
  labs(title = "BlockBench: Frequency vs. Loss (synthetic sequences)", x = "Frequency", y = "Loss") +
  theme_minimal()+
  theme(
    plot.title = element_text(size = 10),
    axis.title.x = element_text(size = 10),
    axis.title.y = element_text(size = 10),
    axis.text.x = element_text(size = 8),
    axis.text.y = element_text(size = 8),
  )
ggsave("tiny_plot.jpg", p, width = 5, height = 2, units = "in", dpi = 300)