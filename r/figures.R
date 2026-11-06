# Somnus figures in a journal / MATLAB-like style (white background, boxed axes, inward ticks, Arial,
# MATLAB default colour order, parula colormap). Captions live in the README, not inside the figures.
# Usage: Rscript r/figures.R [run_dir] [out_dir]

suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(tidyr); library(readr)
  library(patchwork); library(scales); library(jsonlite); library(forcats)
})

args <- commandArgs(trailingOnly = TRUE)
run <- if (length(args) >= 1) args[[1]] else "runs/transformer-fpzcz"
out <- if (length(args) >= 2) args[[2]] else "figures"
ev  <- file.path(run, "eval")
dir.create(out, showWarnings = FALSE, recursive = TRUE)

# MATLAB default line colours and the parula colormap
mlab   <- c(blue = "#0072BD", orange = "#D95319", yellow = "#EDB120", purple = "#7E2F8E",
            green = "#77AC30", cyan = "#4DBEEE", red = "#A2142F")
parula <- c("#352A87", "#0F5CDD", "#1481D6", "#06A4CA", "#2EB7A4", "#87BF77", "#D1BB59", "#FEC832", "#F9FB0E")

stages  <- c("W", "N1", "N2", "N3", "REM")
hyp_ord <- c("N3", "N2", "N1", "REM", "W")
pal <- c(W = mlab[["yellow"]], N1 = mlab[["cyan"]], N2 = mlab[["blue"]], N3 = mlab[["purple"]], REM = mlab[["orange"]])

theme_paper <- function(base = 10) {
  theme_classic(base_size = base, base_family = "Arial") +
    theme(
      panel.border = element_rect(colour = "black", fill = NA, linewidth = 0.5),
      axis.line = element_blank(),
      axis.ticks = element_line(colour = "black", linewidth = 0.4),
      axis.ticks.length = unit(-3, "pt"),
      axis.text = element_text(colour = "black"),
      axis.text.x = element_text(margin = margin(t = 5)),
      axis.text.y = element_text(margin = margin(r = 5)),
      axis.title = element_text(colour = "black"),
      plot.title = element_text(face = "plain", hjust = 0.5, size = rel(1.05)),
      plot.tag = element_text(face = "bold", size = rel(1.25)),
      legend.background = element_rect(colour = "black", linewidth = 0.3, fill = "white"),
      legend.key = element_blank(), legend.title = element_blank(),
      strip.background = element_blank(), strip.text = element_text(colour = "black"),
      plot.background = element_rect(fill = "white", colour = NA),
      plot.margin = margin(8, 10, 8, 8)
    )
}
save <- function(p, name, w, h) {
  ggsave(file.path(out, name), p, width = w, height = h, dpi = 300, device = ragg::agg_png, bg = "white")
  message("wrote ", file.path(out, name))
}
tags <- plot_annotation(tag_levels = "a")
metrics <- fromJSON(file.path(ev, "metrics.json"))

# 1. One held-out night ------------------------------------------------------------------------------
hyp  <- read_csv(file.path(ev, "showcase_hypnogram.csv"), show_col_types = FALSE) |> mutate(hours = epoch * 30 / 3600)
spec <- read_csv(file.path(ev, "showcase_spectrogram.csv"), show_col_types = FALSE) |> mutate(hours = epoch * 30 / 3600)
lim  <- quantile(spec$db, c(0.02, 0.98))

p_spec <- ggplot(spec, aes(hours, freq, fill = pmin(pmax(db, lim[1]), lim[2]))) +
  geom_raster() +
  scale_fill_gradientn(colours = parula, name = "Power (dB)") +
  scale_x_continuous(expand = c(0, 0)) + scale_y_continuous(expand = c(0, 0), breaks = c(0, 10, 20, 30)) +
  labs(x = NULL, y = "Frequency (Hz)") +
  theme_paper() + theme(axis.text.x = element_blank(), legend.position = "right",
                        legend.background = element_blank(), legend.title = element_text(size = rel(0.85)))

steps <- hyp |> filter(true >= 0) |>
  transmute(hours, Expert = match(stages[true + 1], hyp_ord), Somnus = match(stages[pred + 1], hyp_ord) - 0.12) |>
  pivot_longer(-hours, names_to = "scorer", values_to = "y")
p_hyp <- ggplot(steps, aes(hours, y, colour = scorer)) +
  geom_step(linewidth = 0.45) +
  scale_colour_manual(values = c(Expert = "black", Somnus = mlab[["orange"]])) +
  scale_y_continuous(breaks = seq_along(hyp_ord), labels = hyp_ord) +
  scale_x_continuous(expand = c(0, 0)) +
  labs(x = NULL, y = "Stage") +
  theme_paper() + theme(axis.text.x = element_blank(), legend.position = "top", legend.justification = "right",
                        legend.background = element_blank(), legend.margin = margin(0, 0, -6, 0))

probs <- hyp |> select(hours, starts_with("p_")) |>
  pivot_longer(-hours, names_to = "stage", values_to = "p", names_prefix = "p_") |>
  mutate(stage = factor(stage, levels = rev(hyp_ord)))
p_prob <- ggplot(probs, aes(hours, p, fill = stage)) +
  geom_area(position = "fill", linewidth = 0) +
  scale_fill_manual(values = pal) +
  scale_x_continuous(expand = c(0, 0)) + scale_y_continuous(expand = c(0, 0), breaks = c(0, 0.5, 1)) +
  labs(x = "Time (h)", y = "P(stage)") +
  theme_paper() + theme(legend.position = "right", legend.background = element_blank(),
                        legend.key.size = unit(9, "pt"))

save((p_spec / p_hyp / p_prob) + plot_layout(heights = c(1, 1.1, 0.75)) + tags, "hero_night.png", w = 8, h = 6.2)

# 2. Confusion matrix ---------------------------------------------------------------------------------
cm <- read_csv(file.path(ev, "confusion.csv"), show_col_types = FALSE) |>
  group_by(true) |> mutate(recall = n / sum(n)) |> ungroup() |>
  mutate(true = factor(true, levels = rev(stages)), pred = factor(pred, levels = stages))
p_cm <- ggplot(cm, aes(pred, true, fill = recall)) +
  geom_tile(colour = "white", linewidth = 0.6) +
  geom_text(aes(label = sprintf("%.2f", recall), colour = recall > 0.55), size = 3.3, family = "Arial") +
  scale_fill_gradientn(colours = parula, limits = c(0, 1), name = "Recall") +
  scale_colour_manual(values = c(`TRUE` = "black", `FALSE` = "white"), guide = "none") +
  scale_x_discrete(expand = c(0, 0)) + scale_y_discrete(expand = c(0, 0)) +
  coord_equal() +
  labs(x = "Predicted stage", y = "Expert stage") +
  theme_paper() + theme(legend.position = "right", legend.background = element_blank(),
                        legend.title = element_text(size = rel(0.85)), axis.ticks = element_blank())
save(p_cm, "confusion_matrix.png", w = 4.6, h = 3.8)

# 3. Frequency-band attribution -----------------------------------------------------------------------
band_levels <- c("delta (0.5-4 Hz)", "theta (4-8 Hz)", "alpha (8-12 Hz)", "sigma (12-15 Hz)", "beta (15-30 Hz)")
band_short  <- c("Delta", "Theta", "Alpha", "Sigma", "Beta")
bi <- read_csv(file.path(ev, "band_importance.csv"), show_col_types = FALSE) |>
  mutate(stage = factor(stage, levels = rev(stages)),
         band = factor(band, levels = band_levels, labels = band_short))
p_bi <- ggplot(bi, aes(band, stage, fill = prob_drop)) +
  geom_tile(colour = "white", linewidth = 0.6) +
  geom_text(aes(label = sprintf("%.2f", prob_drop)), size = 3.2, family = "Arial") +
  scale_fill_gradient2(low = mlab[["blue"]], mid = "white", high = mlab[["orange"]], midpoint = 0,
                       name = expression(Delta * "P(true stage)")) +
  scale_x_discrete(expand = c(0, 0)) + scale_y_discrete(expand = c(0, 0)) +
  labs(x = "Frequency band removed", y = "Expert stage") +
  theme_paper() + theme(legend.position = "right", legend.background = element_blank(),
                        legend.title = element_text(size = rel(0.85)), axis.ticks = element_blank())
save(p_bi, "band_attribution.png", w = 5.6, h = 3.6)

# 4. Relative band power by stage ---------------------------------------------------------------------
bp <- read_csv(file.path(ev, "band_power.csv"), show_col_types = FALSE) |>
  mutate(stage = factor(stage, levels = stages), band = factor(band, levels = band_levels, labels = band_short))
p_bp <- ggplot(bp, aes(stage, rel_power, fill = stage)) +
  geom_boxplot(outlier.size = 0.2, outlier.alpha = 0.3, linewidth = 0.3, width = 0.65) +
  facet_wrap(~band, nrow = 1, scales = "free_y") +
  scale_fill_manual(values = pal, guide = "none") +
  scale_y_continuous(labels = percent) +
  labs(x = "Sleep stage", y = "Relative power (0.5-30 Hz)") +
  theme_paper()
save(p_bp, "band_power_ridges.png", w = 9, h = 2.9)

# 5. Calibration --------------------------------------------------------------------------------------
rel <- read_csv(file.path(ev, "reliability.csv"), show_col_types = FALSE) |>
  mutate(model = recode(model, uncalibrated = "Uncalibrated", `temperature-scaled` = "Temperature-scaled"))
p_rel <- ggplot(rel, aes(confidence, accuracy, colour = model, shape = model)) +
  geom_abline(linetype = "dashed", colour = "grey40", linewidth = 0.4) +
  geom_line(linewidth = 0.6) + geom_point(size = 1.8) +
  scale_colour_manual(values = c(Uncalibrated = mlab[["blue"]], `Temperature-scaled` = mlab[["orange"]])) +
  scale_shape_manual(values = c(Uncalibrated = 16, `Temperature-scaled` = 1)) +
  scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) + scale_y_continuous(limits = c(0, 1), expand = c(0, 0)) +
  coord_equal() +
  labs(x = "Confidence", y = "Accuracy") +
  theme_paper() + theme(legend.position = "inside", legend.position.inside = c(0.97, 0.03),
                        legend.justification = c(1, 0))
save(p_rel, "calibration.png", w = 3.8, h = 3.8)

# 6. Per-night agreement + training curves ------------------------------------------------------------
pn <- read_csv(file.path(ev, "per_night.csv"), show_col_types = FALSE) |>
  mutate(label = fct_reorder(sprintf("%s (%d%s)", rec_id, age, toupper(substr(sex, 1, 1))), kappa))
f1_long <- pn |> select(label, starts_with("f1_")) |>
  pivot_longer(-label, names_to = "stage", values_to = "f1", names_prefix = "f1_") |>
  mutate(stage = factor(stage, levels = stages))
p_k <- ggplot(pn, aes(kappa, label)) +
  geom_vline(xintercept = metrics$kappa, linetype = "dashed", colour = "grey40", linewidth = 0.4) +
  geom_point(size = 2, colour = mlab[["blue"]]) +
  scale_x_continuous(limits = c(0.5, 0.9)) +
  labs(x = expression("Cohen's " * kappa), y = "Test night (age, sex)") + theme_paper()
p_f1 <- ggplot(f1_long, aes(stage, label, fill = f1)) +
  geom_tile(colour = "white", linewidth = 0.5) +
  geom_text(aes(label = sprintf("%.2f", f1), colour = f1 > 0.55), size = 2.6, family = "Arial") +
  scale_fill_gradientn(colours = parula, limits = c(0, 1), name = "F1") +
  scale_colour_manual(values = c(`TRUE` = "black", `FALSE` = "white"), guide = "none") +
  scale_x_discrete(expand = c(0, 0)) + scale_y_discrete(expand = c(0, 0)) +
  labs(x = "Stage", y = NULL) +
  theme_paper() + theme(axis.text.y = element_blank(), axis.ticks = element_blank(), legend.position = "right",
                        legend.background = element_blank(), legend.title = element_text(size = rel(0.85)))
hist <- read_csv(file.path(run, "history.csv"), show_col_types = FALSE)
p_loss <- ggplot(hist, aes(epoch, train_loss)) +
  geom_line(colour = mlab[["blue"]], linewidth = 0.6) + geom_point(colour = mlab[["blue"]], size = 1.2) +
  labs(x = "Epoch", y = "Training loss") + theme_paper()
val <- hist |> select(epoch, `Cohen's kappa` = val_kappa, `Macro-F1` = val_macro_f1) |> pivot_longer(-epoch)
p_val <- ggplot(val, aes(epoch, value, colour = name, shape = name)) +
  geom_line(linewidth = 0.6) + geom_point(size = 1.4) +
  scale_colour_manual(values = c(mlab[["orange"]], mlab[["yellow"]]) |> unname()) +
  scale_shape_manual(values = c(16, 17)) +
  labs(x = "Epoch", y = "Validation score") +
  theme_paper() + theme(legend.position = "inside", legend.position.inside = c(0.97, 0.03),
                        legend.justification = c(1, 0))
p_nights <- ((p_k | p_f1) + plot_layout(widths = c(1, 1))) / (p_loss | p_val) + plot_layout(heights = c(1.5, 1)) + tags
save(p_nights, "per_night_and_training.png", w = 8.5, h = 7)
