#!/usr/bin/env Rscript

# Load required libraries
library(ggplot2)
library(dplyr)

# Read command line arguments: method, max_steps, seed
args <- commandArgs(trailingOnly = TRUE)
method_choice <- if (length(args) > 0) args[1] else "loss"
max_steps <- if (length(args) > 1) args[2] else "5000"
seed <- if (length(args) > 2) args[3] else "1234"
steps_k <- paste0(as.integer(max_steps) %/% 1000, "k")

# Read the data
if (seed == "all") {
  att <- read.csv("/tmp/att_appendix_all.csv")
  atu <- read.csv("/tmp/atu_appendix_all.csv")
} else {
  att <- read.csv(sprintf("/tmp/att_appendix_steps%s_seed%s.csv", steps_k, seed))
  atu <- read.csv(sprintf("/tmp/atu_appendix_steps%s_seed%s.csv", steps_k, seed))
}

# Select only the common columns we need
att <- att %>% select(method, doc_id, delta)
atu <- atu %>% select(method, doc_id, delta)

# Add group column
att$group <- "Contaminated docs"
atu$group <- "Uncontaminated docs"

# Concatenate
data <- rbind(att, atu)

# Filter to only the chosen method
data <- data %>% filter(method == method_choice)

# Create ECDF plot showing both ATT and ATU
p_ecdf <- ggplot(data, aes(x = delta, color = group)) +
  stat_ecdf(linewidth = 1.5, geom = "step") +
  geom_vline(xintercept = 0, linetype = "dashed", color = "black", alpha = 0.7) +
  scale_color_manual(
    values = c("Contaminated docs" = "#d95f02", "Uncontaminated docs" = "#7570b3"),
    limits = c("Uncontaminated docs", "Contaminated docs"),
    labels = c(
      "Uncontaminated docs" = expression("Uncontaminated docs " * delta[i]^{D==0}),
      "Contaminated docs" = expression("Contaminated docs " * delta[i]^{D==1})
    )
  ) +
  labs(
    x = expression("Document-level effect " * delta[i]),
    y = "Cumulative Probability",
    color = ""
  ) +
  scale_x_continuous(limits = c(-0.2, 0.2)) +
  theme_minimal() +
  theme(
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    legend.position = "right",
    legend.text = element_text(size = 24),
    axis.ticks = element_line(linewidth = 2.5),
    axis.ticks.length = unit(0.5, "cm"),
    axis.text = element_text(size = 26),
    axis.title.x = element_text(size = 28, color = "gray20", margin = margin(t = 10)),
    axis.title.y = element_text(size = 28, margin = margin(r = 10), vjust = 0, hjust = 1),
    plot.caption = element_text(size = 10, color = "gray40", hjust = 0.5, margin = margin(t = 5)),
    plot.margin = margin(10, 10, 10, 10)
  )

# Save the ECDF plot
output_file <- sprintf("figures/delta_0_appendix_ecdf_%s_steps%s_seed%s.png", method_choice, steps_k, seed)
ggsave(output_file, plot = p_ecdf, width = 16, height = 4.5, dpi = 300, bg = "white")

cat(sprintf("\nECDF plot saved to %s\n", output_file))
cat(sprintf("Method: %s\n", method_choice))
cat(sprintf("Total rows for this method: %d\n", nrow(data)))
