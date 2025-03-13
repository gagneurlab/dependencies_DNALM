source('data/ecoli_dms_processed_data/probs.R')


library(data.table)
library(ggplot2)
library(cowplot)

DT <- data.table(
  scored   = c(rep("Pairs in anti-parallel\ndependencies", length(antidiag)),
               rep("All pairs", length(all))),
  prob = c(antidiag, all)
)

# 2. Plot with ggplot2. 
#    Mirroring your original plotnine aesthetics:
ggplot(DT, aes(x = scored, y = prob)) +
  geom_boxplot(fill = "lightblue", alpha=.2) +
  scale_y_log10() +
  annotation_logticks(sides = "l") +
  labs(x = "", y = "Base-pair probability") +
  theme_cowplot()

DT <- data.table(
  scored = c(
    rep("Pairs in anti-parallel\ndependencies", length(antidiag)),
    rep("All pairs", length(all))
  ),
  prob = c(antidiag, all)
)

count_DT <- DT[, .N, by = scored]
count_DT

new_labels <- setNames(
  paste0(count_DT$scored, "\nN=", count_DT$N),
  count_DT$scored
)


ggplot(DT, aes(x = scored, y = prob)) +
  geom_boxplot(fill = "lightblue", alpha=.2) +
  scale_y_log10() +
  annotation_logticks(sides = "l") +
  labs(x = "", y = "Base-pair Probability") +
  scale_x_discrete(labels = new_labels) +
  theme_cowplot()



source('data/ecoli_dms_processed_data/dms_react.R')



DT <- data.table(
  scored   = c(rep("Nucleotides in\nanti-parallel dependencies", length(antidiag)),
               rep("Other", length(others))),
  DMS_reac = c(antidiag, others)
)
count_DT <- DT[, .N, by = scored]
count_DT
# 2. Plot with ggplot2. 
#    Mirroring your original plotnine aesthetics:
ggplot(DT, aes(x = scored, y = DMS_reac)) +
  geom_boxplot(fill = "lightblue", alpha=.2) +
  scale_y_log10() +
  annotation_logticks(sides = "l") +
  labs(x = "", y = "DMS-MaPseq mutation frequency") +
  theme_cowplot()

ggplot(DT, aes(x = scored, y = DMS_reac)) +
  geom_boxplot(fill = "lightblue", alpha=.2,outlier.shape = NA) +
  labs(x = "", y = "DMS-MaPseq Mutation Frequency") +
  theme_cowplot() +
  ylim(c(0,.07))

