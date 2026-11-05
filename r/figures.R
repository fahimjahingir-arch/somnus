# Somnus figures: ggplot2 + patchwork, rendered with ragg.
# Usage: Rscript r/figures.R [run_dir] [out_dir]

suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(tidyr); library(readr)
  library(patchwork); library(ggridges); library(scales); library(jsonlite); library(forcats)
})

args <- commandArgs(trailingOnly = TRUE)
run <- if (length(args) >= 1) args[[1]] else "runs/transformer-fpzcz"
out <- if (length(args) >= 2) args[[2]] else "figures"
ev  <- file.path(run, "eval")
dir.create(out, showWarnings = FALSE, recursive = TRUE)

stages  <- c("W", "N1", "N2", "N3", "REM")
hyp_ord <- c("N3", "N2", "N1", "REM", "W")           # clinical hypnogram order, deepest at bottom
pal <- c(W = "#F6C85F", N1 = "#9DD9D2", N2 = "#4F9DDE", N3 = "#2B3A8C", REM = "#EF6F6C")
bg <- "#0E1117"; fg <- "#E8EAF0"; muted <- "#8B93A7"; grid <- "#222838"

theme_somnus <- function(base = 13) {
  theme_minimal(base_size = base, base_family = "sans") +
    theme(
      plot.background  = element_rect(fill = bg, colour = NA),
      panel.background = element_rect(fill = bg, colour = NA),
      panel.grid.major = element_line(colour = grid, linewidth = 0.3),
      panel.grid.minor = element_blank(),
      text = element_text(colour = fg), axis.text = element_text(colour = muted),
      plot.title = element_text(face = "bold", size = rel(1.35)),
      plot.subtitle = element_text(colour = muted, margin = margin(b = 10)),
      plot.caption = element_text(colour = muted, size = rel(0.75)),
      legend.position = "top", legend.justification = "left",
      strip.text = element_text(colour = fg, face = "bold"),
      plot.margin = margin(16, 18, 12, 14)
    )
}
save <- function(p, name, w = 12, h = 7) {
  ggsave(file.path(out, name), p, width = w, height = h, dpi = 170, device = ragg::agg_png, bg = bg)
  message("wrote ", file.path(out, name))
}

metrics <- fromJSON(file.path(ev, "metrics.json"))

# 1. Hero: one held-out night - spectrogram, expert vs model hypnogram, model probabilities ----------
hyp  <- read_csv(file.path(ev, "showcase_hypnogram.csv"), show_col_types = FALSE) |>
  mutate(hours = epoch * 30 / 3600)
spec <- read_csv(file.path(ev, "showcase_spectrogram.csv"), show_col_types = FALSE) |>
  mutate(hours = epoch * 30 / 3600)
showcase_id <- hyp$rec_id[1]
night_kappa <- read_csv(file.path(ev, "per_night.csv"), show_col_types = FALSE) |>
  filter(rec_id == showcase_id) |> pull(kappa)

lim <- quantile(spec$db, c(0.02, 0.98))
p_spec <- ggplot(spec, aes(hours, freq, fill = pmin(pmax(db, lim[1]), lim[2]))) +
  geom_raster(interpolate = TRUE) +
  scale_fill_viridis_c(option = "magma", guide = "none") +
  scale_x_continuous(expand = c(0, 0)) + scale_y_continuous(expand = c(0, 0), breaks = c(4, 8, 12, 15, 30)) +
  labs(y = "Hz", x = NULL,
       title = "One night, never seen during training",
       subtitle = sprintf(paste0("EEG spectrogram (Fpz-Cz), expert scoring vs Somnus, and calibrated stage probabilities",
                                 "  |  night %s, Cohen's kappa = %.2f"), showcase_id, night_kappa)) +
  theme_somnus() + theme(axis.text.x = element_blank())

steps <- hyp |>
  transmute(hours, Expert = stages[true + 1], Somnus = stages[pred + 1]) |>
  filter(!is.na(Expert)) |>
  pivot_longer(c(Expert, Somnus), names_to = "scorer", values_to = "stage") |>
  mutate(stage = factor(stage, levels = hyp_ord), y = as.integer(stage))
p_hyp <- ggplot(steps, aes(hours, y)) +
  geom_step(colour = muted, linewidth = 0.35) +
  geom_point(aes(colour = stage), size = 0.7, shape = 15) +
  facet_wrap(~scorer, ncol = 1) +
  scale_colour_manual(values = pal, guide = "none") +
  scale_y_continuous(breaks = seq_along(hyp_ord), labels = hyp_ord) +
  scale_x_continuous(expand = c(0, 0)) +
  labs(x = NULL, y = NULL) + theme_somnus() +
  theme(axis.text.x = element_blank(), strip.text = element_text(hjust = 0))

probs <- hyp |>
  select(hours, starts_with("p_")) |>
  pivot_longer(-hours, names_to = "stage", values_to = "p", names_prefix = "p_") |>
  mutate(stage = factor(stage, levels = rev(hyp_ord)))
p_prob <- ggplot(probs, aes(hours, p, fill = stage)) +
  geom_area(position = "fill", linewidth = 0) +
  scale_fill_manual(values = pal, name = NULL) +
  scale_x_continuous(expand = c(0, 0), labels = \(h) paste0(h, "h")) +
  scale_y_continuous(labels = percent, breaks = c(0, .5, 1)) +
  labs(x = "time since recording window start", y = "P(stage)") + theme_somnus() +
  theme(legend.position = "bottom")

hero <- (p_spec / p_hyp / p_prob) + plot_layout(heights = c(1.1, 1.6, 0.9)) &
  theme(plot.background = element_rect(fill = bg, colour = NA))
save(hero, "hero_night.png", w = 13, h = 9)

# 2. Confusion matrix (row-normalised = recall) -----------------------------------------------------
cm <- read_csv(file.path(ev, "confusion.csv"), show_col_types = FALSE) |>
  group_by(true) |> mutate(recall = n / sum(n)) |> ungroup() |>
  mutate(true = factor(true, levels = rev(stages)), pred = factor(pred, levels = stages))
p_cm <- ggplot(cm, aes(pred, true, fill = recall)) +
  geom_tile(colour = bg, linewidth = 1.2) +
  geom_text(aes(label = sprintf("%.0f%%\n%s", 100 * recall, comma(n)),
                colour = recall > 0.5), size = 3.6, lineheight = 0.9) +
  scale_fill_viridis_c(option = "mako", limits = c(0, 1), labels = percent, name = "row %") +
  scale_colour_manual(values = c(`TRUE` = bg, `FALSE` = fg), guide = "none") +
  coord_equal() +
  labs(x = "Somnus prediction", y = "Expert label",
       title = "Where the model agrees and disagrees",
       subtitle = sprintf("%s held-out epochs · accuracy %.1f%% · macro-F1 %.3f · kappa %.3f",
                          comma(metrics$n_test_epochs), 100 * metrics$accuracy, metrics$macro_f1, metrics$kappa)) +
  theme_somnus() + theme(panel.grid = element_blank(), legend.position = "right")
save(p_cm, "confusion_matrix.png", w = 9, h = 7.5)

# 3. Frequency-band attribution heatmap -------------------------------------------------------------
band_levels <- c("delta (0.5-4 Hz)", "theta (4-8 Hz)", "alpha (8-12 Hz)", "sigma (12-15 Hz)", "beta (15-30 Hz)")
bi <- read_csv(file.path(ev, "band_importance.csv"), show_col_types = FALSE) |>
  mutate(stage = factor(stage, levels = rev(stages)),
         band = factor(band, levels = band_levels))
p_bi <- ggplot(bi, aes(band, stage, fill = prob_drop)) +
  geom_tile(colour = bg, linewidth = 1.2) +
  geom_text(aes(label = sprintf("%+.2f", prob_drop)), colour = fg, size = 3.8) +
  scale_fill_gradient2(low = "#3B6FB6", mid = bg, high = "#F28E2B", midpoint = 0, name = "Δ P(true stage)") +
  labs(x = NULL, y = NULL, title = "What the network listens to",
       subtitle = paste0("Drop in probability of the correct stage when one EEG frequency band is filtered out of held-out nights.\n",
                         "Deep sleep (N3) collapses without delta slow waves, N2 needs sigma (sleep spindles), REM leans on theta.")) +
  theme_somnus() + theme(panel.grid = element_blank(), legend.position = "right")
save(p_bi, "band_attribution.png", w = 11, h = 6)

# 4. Ridgelines: relative band power by stage -------------------------------------------------------
bp <- read_csv(file.path(ev, "band_power.csv"), show_col_types = FALSE) |>
  mutate(stage = factor(stage, levels = rev(stages)), band = fct_inorder(band))
p_bp <- ggplot(bp, aes(rel_power, stage, fill = stage)) +
  geom_density_ridges(colour = bg, alpha = 0.9, scale = 1.6, rel_min_height = 0.01, linewidth = 0.3) +
  facet_wrap(~band, nrow = 1, scales = "free_x") +
  scale_fill_manual(values = pal, guide = "none") +
  scale_x_continuous(labels = percent, n.breaks = 3) +
  labs(x = "share of 0.5-30 Hz power", y = NULL, title = "The EEG signature of each sleep stage",
       subtitle = "Distribution of relative band power across every held-out 30-second epoch") +
  theme_somnus()
save(p_bp, "band_power_ridges.png", w = 14, h = 6)

# 5. Calibration ------------------------------------------------------------------------------------
rel <- read_csv(file.path(ev, "reliability.csv"), show_col_types = FALSE)
p_rel <- ggplot(rel, aes(confidence, accuracy, colour = model)) +
  geom_abline(linetype = "dashed", colour = muted) +
  geom_line(linewidth = 0.9) + geom_point(aes(size = n), alpha = 0.85) +
  scale_colour_manual(values = c(uncalibrated = "#EF6F6C", `temperature-scaled` = "#9DD9D2"), name = NULL) +
  scale_size_area(max_size = 6, guide = "none") +
  scale_x_continuous(labels = percent, limits = c(0, 1)) + scale_y_continuous(labels = percent, limits = c(0, 1)) +
  coord_equal() +
  labs(x = "model confidence", y = "observed accuracy", title = "Can you trust its confidence?",
       subtitle = sprintf(paste0("Points above the diagonal = slightly under-confident (a side effect of label smoothing).\n",
                                 "ECE %.3f raw vs %.3f after temperature scaling (T = %.2f fit on validation) - it did not transfer."),
                          metrics$ece_uncalibrated, metrics$ece_calibrated, metrics$temperature)) +
  theme_somnus()
save(p_rel, "calibration.png", w = 8.5, h = 8)

# 6. Per-night agreement + training curve -----------------------------------------------------------
pn <- read_csv(file.path(ev, "per_night.csv"), show_col_types = FALSE) |>
  mutate(rec_id = fct_reorder(sprintf("%s  (%d %s)", rec_id, age, toupper(substr(sex, 1, 1))), kappa))
f1_long <- pn |> select(rec_id, starts_with("f1_")) |>
  pivot_longer(-rec_id, names_to = "stage", values_to = "f1", names_prefix = "f1_") |>
  mutate(stage = factor(stage, levels = stages))
p_k <- ggplot(pn, aes(kappa, rec_id)) +
  geom_segment(aes(x = 0, xend = kappa, yend = rec_id), colour = grid, linewidth = 2) +
  geom_point(size = 4, colour = "#9DD9D2") +
  geom_vline(xintercept = 0.8, linetype = "dotted", colour = muted) +
  annotate("text", x = 0.81, y = 1.5, label = "typical human\ninter-scorer\nagreement", hjust = 0,
           size = 3, colour = muted, lineheight = 0.9) +
  scale_x_continuous(limits = c(0, 1)) +
  labs(x = "Cohen's kappa", y = NULL, title = "Every held-out night",
       subtitle = "label = recording (age, sex)") +
  theme_somnus()
p_f1 <- ggplot(f1_long, aes(stage, rec_id, fill = f1)) +
  geom_tile(colour = bg, linewidth = 1) +
  geom_text(aes(label = sprintf("%.2f", f1)), size = 3, colour = bg) +
  scale_fill_viridis_c(option = "mako", limits = c(0, 1), begin = 0.25, name = "F1") +
  labs(x = NULL, y = NULL, title = "Per-stage F1") + theme_somnus() +
  theme(axis.text.y = element_blank(), panel.grid = element_blank(), legend.position = "right")
hist <- read_csv(file.path(run, "history.csv"), show_col_types = FALSE) |>
  select(epoch, `train loss` = train_loss, `val kappa` = val_kappa, `val macro-F1` = val_macro_f1) |>
  pivot_longer(-epoch)
p_hist <- ggplot(hist, aes(epoch, value, colour = name)) +
  geom_line(linewidth = 1) + geom_point(size = 1.4) +
  facet_wrap(~name, scales = "free_y", nrow = 1) +
  scale_colour_manual(values = c("#EF6F6C", "#4F9DDE", "#9DD9D2"), guide = "none") +
  labs(x = "epoch", y = NULL, title = "Training (validation subjects only, never test)") + theme_somnus()
p_nights <- ((p_k | p_f1) + plot_layout(widths = c(1.3, 1))) / p_hist + plot_layout(heights = c(1.4, 1))
save(p_nights, "per_night_and_training.png", w = 13, h = 9)
