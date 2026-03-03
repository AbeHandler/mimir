#!/usr/bin/env Rscript

# Load required libraries
library(ggplot2)
library(dplyr)

# Read command line arguments
args <- commandArgs(trailingOnly = TRUE)
method_choice <- if (length(args) > 0) args[1] else "loss"

# Read the data
att <- read.csv("/tmp/att.csv")
atu <- read.csv("/tmp/atu.csv")

# Add group column
att$group <- "treated"
atu$group <- "untreated"

# Concatenate
data <- rbind(att, atu)

# Filter to only the chosen method
data <- data %>% filter(method == method_choice)

# Rename method for better labels
method_label <- ifelse(method_choice == "min_k", "Min-K%", "LOSS")

# Create density plot with ATT vs ATU shown
p <- ggplot(data, aes(x = delta, fill = group, color = group)) +
  geom_density(alpha = 0.5, linewidth = 1) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "black", alpha = 0.7) +
  scale_fill_manual(
    values = c("treated" = "#E69F00", "untreated" = "#56B4E9"),
    labels = c(
      "treated" = expression("Treated/In-sample (" * delta^{D==1} * ")"),
      "untreated" = expression("Untreated/Out-of-sample (" * delta^{D==0} * ")")
    ),
    limits = c("untreated", "treated")
  ) +
  scale_color_manual(
    values = c("treated" = "#E69F00", "untreated" = "#56B4E9"),
    labels = c(
      "treated" = expression("Treated/In-sample (" * delta^{D==1} * ")"),
      "untreated" = expression("Untreated/Out-of-sample (" * delta^{D==0} * ")")
    ),
    limits = c("untreated", "treated")
  ) +
  scale_x_continuous(limits = c(-0.75, 0.75)) +
  labs(
    x = expression("Treatment effect " * delta),
    y = "Density",
    fill = "",
    color = ""
  ) +
  theme_minimal() +
  theme(
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    legend.position = "top",
    axis.ticks = element_line(linewidth = 1.5),
    axis.ticks.length = unit(0.3, "cm"),
    axis.text = element_text(size = 14),
    axis.title.x = element_text(size = 16, color = "gray20"),
    axis.title.y = element_text(size = 16),
    plot.margin = margin(10, 10, 40, 10)
  )

# Add custom text annotations below the plot using grid
library(grid)
library(gridExtra)

# Create text grobs with different sizes
text1 <- textGrob(
  expression("Treatment effect " * delta),
  gp = gpar(fontsize = 16, col = "gray20")
)
text2 <- textGrob(
  paste0("(", method_label, " metric, GPT-2 trained from scratch)"),
  gp = gpar(fontsize = 11, col = "gray30")
)
text3 <- textGrob(
  "Axis shows data within \u00B12 standard deviations of the mean",
  gp = gpar(fontsize = 9, col = "gray50")
)

# Arrange plot with text below
p_with_text <- arrangeGrob(
  p,
  text1,
  text2,
  text3,
  ncol = 1,
  heights = c(10, 0.4, 0.3, 0.3)
)

# Save the density plot
ggsave("/tmp/plot.png", plot = p_with_text, width = 8, height = 4.5, dpi = 300)

cat("Density plot saved to /tmp/plot.png\n")

# Create ECDF plot
p_ecdf <- ggplot(data, aes(x = delta, color = group)) +
  stat_ecdf(linewidth = 1.5, geom = "step") +
  geom_vline(xintercept = 0, linetype = "dashed", color = "black", alpha = 0.7) +
  scale_color_manual(
    values = c("treated" = "#E69F00", "untreated" = "#56B4E9"),
    labels = c(
      "treated" = expression("Treated/In-sample (" * delta^{D==1} * ")"),
      "untreated" = expression("Untreated/Out-of-sample (" * delta^{D==0} * ")")
    ),
    limits = c("untreated", "treated")
  ) +
  scale_x_continuous(limits = c(-0.75, 0.75)) +
  labs(
    x = expression("Treatment effect " * delta),
    y = "Empirical Cumulative Probability",
    color = ""
  ) +
  theme_minimal() +
  theme(
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    legend.position = "top",
    axis.ticks = element_line(linewidth = 1.5),
    axis.ticks.length = unit(0.3, "cm"),
    axis.text = element_text(size = 14),
    axis.title.x = element_text(size = 16, color = "gray20"),
    axis.title.y = element_text(size = 16),
    plot.margin = margin(10, 10, 40, 10)
  )

# Create text grobs for ECDF plot
text1_ecdf <- textGrob(
  expression("Treatment effect " * delta),
  gp = gpar(fontsize = 16, col = "gray20")
)
text2_ecdf <- textGrob(
  paste0("(", method_label, " metric, GPT-2 trained from scratch)"),
  gp = gpar(fontsize = 11, col = "gray30")
)
# Arrange ECDF plot with text below (no SD comment for ECDF)
p_ecdf_with_text <- arrangeGrob(
  p_ecdf,
  text1_ecdf,
  text2_ecdf,
  ncol = 1,
  heights = c(10, 0.4, 0.3)
)

# Save the ECDF plot
ggsave("/tmp/plot_ecdf.png", plot = p_ecdf_with_text, width = 8, height = 4.5, dpi = 300)

cat("ECDF plot saved to /tmp/plot_ecdf.png\n")
cat(sprintf("Method: %s\n", method_label))
cat(sprintf("Total rows ATT: %d\n", nrow(att)))
cat(sprintf("Total rows ATU: %d\n", nrow(atu)))
cat(sprintf("Total combined: %d\n", nrow(data)))
