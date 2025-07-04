library(tidyverse)

# Read both files
loss <- read_csv("loss.csv") %>% mutate(method = if_else(method == "loss", "loss/ref", method))
min_k <- read_csv("min_k.csv")
dcpdd <- read_csv("dcpdd.csv")

# Combine
combined <- bind_rows(loss, min_k, dcpdd)

# Extract numeric lower bounds for ordering
combined <- combined %>%
  mutate(size_bin = factor(size_bin, levels = combined %>%
                             distinct(size_bin) %>%
                             mutate(bin_order = as.numeric(str_extract(size_bin, "(?<=\\()[0-9]+"))) %>%
                             arrange(bin_order) %>%
                             pull(size_bin)))

# Plot
ggplot(combined, aes(x = size_bin, y = delta_mean, color = method, group = method)) +
  geom_line() +
  geom_point() +
  labs(
    x = "Number of similar documents in news ecosystem", y = "ATE",
    title = "ATE vs. Text Reuse by Method"
  ) +
  theme_minimal()

ggsave("tmp.pdf")