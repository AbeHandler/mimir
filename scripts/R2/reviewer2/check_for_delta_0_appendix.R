#!/usr/bin/env Rscript

# Load required libraries
library(ggplot2)
library(dplyr)

# Read the data
blocks <- read.csv('csvs/confounddataset/pythia-45m_lr1e-3_steps5k_seed1234_interleave0.02_uncontaminated_insample_regular_training_data.all_shards.csv.gz') %>%
  rename(blocks = score)

noblocks <- read.csv('csvs/confounddataset/pythia-45m_lr1e-3_steps5k_seed1234_uncontaminated_insample_regular_training_data.all_shards.csv.gz') %>%
  rename(noblocks = score)

# Filter for members only
blocks <- blocks %>% filter(membership == "member")
noblocks <- noblocks %>% filter(membership == "member")

# Merge on method and doc_id
both <- blocks %>%
  inner_join(noblocks, by = c("method", "doc_id"))

# Calculate delta
both$delta <- both$blocks - both$noblocks

# Print mean delta by method (matching Python output)
cat("\nMean delta by method:\n")
mean_by_method <- both %>%
  group_by(method) %>%
  summarise(delta = mean(delta)) %>%
  arrange(method)

print(mean_by_method, n = Inf)

# Create ECDF plot
p_ecdf <- ggplot(both, aes(x = delta, color = method)) +
  stat_ecdf(linewidth = 1.5, geom = "step") +
  geom_vline(xintercept = 0, linetype = "dashed", color = "black", alpha = 0.7) +
  labs(
    x = expression("Effect " * delta * " (blocks - noblocks). Expect delta = 0."),
    y = "Empirical Cumulative Probability",
    caption = "Pythia-45m: contaminated vs uncontaminated training data (members only)",
    color = "Method"
  ) +
  theme_minimal() +
  theme(
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    legend.position = "top",
    axis.ticks = element_line(linewidth = 1.5),
    axis.ticks.length = unit(0.3, "cm"),
    axis.text = element_text(size = 14),
    axis.title.x = element_text(size = 16, color = "gray20", margin = margin(t = 10)),
    axis.title.y = element_text(size = 16),
    plot.caption = element_text(size = 10, color = "gray40", hjust = 0.5, margin = margin(t = 5)),
    plot.margin = margin(10, 10, 10, 10)
  )

# Save the ECDF plot
ggsave("figures/delta_0_appendix_ecdf.png", plot = p_ecdf, width = 8, height = 4.5, dpi = 300, bg = "white")

cat("\nECDF plot saved to figures/delta_0_appendix_ecdf.png\n")
cat(sprintf("Total rows: %d\n", nrow(both)))
cat(sprintf("Unique methods: %d\n", n_distinct(both$method)))
