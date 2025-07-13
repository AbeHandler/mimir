library(tidyverse)

# Read both files
loss <- read_csv("loss.csv") %>% mutate(method = if_else(method == "loss", "loss/ref", method))
min_k <- read_csv("min_k.csv")
dcpdd <- read_csv("dcpdd.csv")
neighborhood <- read_csv("neighborhood.csv")
zlib <- read_csv("zlib.csv")
doc <- read_csv("doc.csv")

# Combine
combined <- bind_rows(loss, min_k, dcpdd, neighborhood, zlib, doc)

combined <- combined %>%
  mutate(method = str_to_upper(str_replace_all(method, "_", " ")))

# Extract numeric lower bounds for ordering
combined <- combined %>%
  mutate(size_bin = factor(size_bin, levels = combined %>%
                             distinct(size_bin) %>%
                             mutate(bin_order = as.numeric(str_extract(size_bin, "(?<=\\()[0-9]+"))) %>%
                             arrange(bin_order) %>%
                             pull(size_bin)))

# Plot with 5 columns, 1 row
ggplot(combined, aes(x = size_bin, y = delta_mean, group = method)) +
  geom_line(color = "grey40", size = 1.2) +
  geom_point(color = "grey40", size = 2) +
  facet_wrap(~ method, nrow = 1, scales = "free_y") +
  labs(
    x = "Number of similar documents in news ecosystem", 
    y = "ATT",
    title = "ATT by Number of Similar Documents by Method"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    strip.text = element_text(face = "bold"), 
    legend.position = "none",  # Remove legend since method is shown in facet titles
    panel.background = element_rect(fill = "white", color = NA),
    plot.background = element_rect(fill = "white", color = NA),
    plot.margin = margin(0, 0, 0, 0, "mm")
  )

ggsave("plots/bymethod_5col.png", width = 15, height = 4)  # Adjust width/height as needed