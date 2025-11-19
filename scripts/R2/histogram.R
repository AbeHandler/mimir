library(ggplot2)

# Load data
ATTloss <- read.csv("data/interim/bothbins/ATTsfromR1/loss.csv")$delta
ATUloss <- read.csv("data/interim/bothbins/loss.granular.csv")$delta

# Combine into dataframe
df <- data.frame(
  delta = c(ATTloss, ATUloss),
  group = c(rep("ATT", length(ATTloss)), rep("ATU", length(ATUloss)))
)

# Find 95% range (2.5th to 97.5th percentile)
x_lower <- quantile(df$delta, 0.025)
x_upper <- quantile(df$delta, 0.975)

# Plot with limited x-axis
p = ggplot(df, aes(x = delta, fill = group)) +
  geom_histogram(alpha = 0.3, position = "identity", bins = 60) +
  xlim(x_lower, x_upper) +
  labs(title = "Distribution of Loss: ATT vs ATU (95% of data)",
       x = "Delta (Loss)",
       y = "Count",
       fill = "Group") +
  theme_minimal()

ggsave("figures/att_atu_histogram.png", plot = p, width = 10, height = 6, dpi = 300, bg='white')