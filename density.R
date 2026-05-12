#!/usr/bin/env Rscript

# Load required libraries
library(ggplot2)
library(dplyr)

# Read command line arguments
args <- commandArgs(trailingOnly = TRUE)
method_choice <- if (length(args) > 0) args[1] else "loss"

# --- colors from ~/dolma/config/plots.ini ---
treated_color <- "#1b9e77"    # color1
untreated_color <- "#d95f02"  # color2

# --- axis styling from ~/dolma/config/plots.ini [axis] ---
tick_length    <- 8     # pt
line_width     <- 0.5
axis_text_size <- 17
base_size      <- 16
title_size     <- 19

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
    values = c("treated" = treated_color, "untreated" = untreated_color),
    labels = c(
      "treated" = "Treated Documents",
      "untreated" = "Untreated Documents"
    ),
    limits = c("untreated", "treated")
  ) +
  scale_color_manual(
    values = c("treated" = treated_color, "untreated" = untreated_color),
    labels = c(
      "treated" = "Treated Documents",
      "untreated" = "Untreated Documents"
    ),
    limits = c("untreated", "treated")
  ) +
  scale_x_continuous(limits = c(-0.75, 0.75)) +
  labs(
    x = "Causal effect of treatment",
    y = "Density",
    fill = "",
    color = ""
  ) +
  theme_minimal(base_size = base_size) +
  theme(
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    legend.position = "top",
    axis.ticks = element_line(linewidth = line_width),
    axis.ticks.length = unit(tick_length, "pt"),
    axis.text = element_text(size = axis_text_size),
    axis.title.x = element_text(size = title_size, color = "gray20", margin = margin(t = 10)),
    axis.title.y = element_text(size = title_size),
    plot.margin = margin(10, 10, 10, 10)
  )

# Save the density plot
ggsave("/tmp/plot.png", plot = p, width = 8, height = 4.5, dpi = 300)

cat("Density plot saved to /tmp/plot.png\n")

# Create ECDF plot
p_ecdf <- ggplot(data, aes(x = delta, color = group)) +
  stat_ecdf(linewidth = 1.5, geom = "step") +
  geom_vline(xintercept = 0, linetype = "dashed", color = "black", alpha = 0.7) +
  scale_color_manual(
    values = c("treated" = treated_color, "untreated" = untreated_color),
    labels = c(
      "treated" = "Treated Documents",
      "untreated" = "Untreated Documents"
    ),
    limits = c("untreated", "treated")
  ) +
  scale_x_continuous(limits = c(-0.75, 0.75)) +
  labs(
    x = "Causal effect of treatment",
    y = "Empirical Cumulative Probability",
    color = ""
  ) +
  theme_minimal(base_size = base_size) +
  theme(
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    legend.position = "top",
    axis.ticks = element_line(linewidth = line_width),
    axis.ticks.length = unit(tick_length, "pt"),
    axis.text = element_text(size = axis_text_size),
    axis.title.x = element_text(size = title_size, color = "gray20", margin = margin(t = 10)),
    axis.title.y = element_text(size = title_size),
    plot.margin = margin(10, 10, 10, 10)
  )

# Save the ECDF plot
ggsave("/tmp/plot_ecdf.png", plot = p_ecdf, width = 8, height = 4.5, dpi = 300)

cat("ECDF plot saved to /tmp/plot_ecdf.png\n")
cat(sprintf("Method: %s\n", method_label))
cat(sprintf("Total rows ATT: %d\n", nrow(att)))
cat(sprintf("Total rows ATU: %d\n", nrow(atu)))
cat(sprintf("Total combined: %d\n", nrow(data)))
