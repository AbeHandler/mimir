library(tidyverse)

# Read both files
loss <- read_csv("data/interim/bothbins/loss.csv") %>% mutate(method = if_else(method == "loss", "loss/ref", method))
min_k <- read_csv("data/interim/bothbins/min_k.csv")
zlib <- read_csv("data/interim/bothbins/zlib.csv")

#doc <- read_csv("doc.csv")
#dcpdd <- read_csv("dcpdd.csv")
#neighborhood <- read_csv("NE.csv")

# Combine
combined <- bind_rows(loss, min_k, zlib)

combined <- combined %>%
  mutate(method = str_to_upper(str_replace_all(method, "_", " ")))

# Extract numeric lower bounds for ordering
combined <- combined %>%
  mutate(size_bin = factor(size_bin, levels = combined %>%
                             distinct(size_bin) %>%
                             mutate(bin_order = as.numeric(str_extract(size_bin, "(?<=\\()[0-9]+"))) %>%
                             arrange(bin_order) %>%
                             pull(size_bin)))

# Compute max absolute delta_mean per method
facet_limits <- combined %>%
  group_by(method) %>%
  summarize(max_abs = max(abs(delta_mean), na.rm = TRUE), .groups = "drop")

# Add invisible points to enforce symmetric y-limits per facet
combined <- combined %>%
  left_join(facet_limits, by = "method") %>%
  rowwise() %>%
  mutate(
    delta_mean_upper = max_abs,
    delta_mean_lower = -max_abs
  )

ggplot(combined, aes(x = size_bin, y = delta_mean, group = method)) +
  geom_line(color = "grey40", size = 1.2) +
  geom_point(color = "grey40", size = 2) +
  geom_hline(yintercept = 0, color = "red", linetype = "solid") +
  # invisible points to force symmetric limits per facet
  geom_point(aes(y = delta_mean_upper), alpha = 0) +
  geom_point(aes(y = delta_mean_lower), alpha = 0) +
  facet_wrap(~ method, nrow = 1, scales = "free_y") +
  labs(
    x = "Number of similar documents in news ecosystem",
    y = "ATU"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 24, face = "bold"),       # big title
    axis.title = element_text(size = 18),                      # big axis labels
    axis.text = element_text(size = 14),                       # big tick labels
    strip.text = element_text(size = 18, face = "bold"),       # big facet labels
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "none",
    panel.background = element_rect(fill = "white", color = NA),
    plot.background = element_rect(fill = "white", color = NA)
  )
ggsave("plots/both_bins_R2.png", width = 15, height = 4)  # Adjust width/height as needed