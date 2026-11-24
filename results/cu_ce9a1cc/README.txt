OPERA results
=============

visualize_results.ipynb   tables: mean +/- std, % improvement vs baseline, averages, win counts
results/
  per_seed_results.csv    every individual run (setting x backbone x method x seed x condition)
  forecast_clean.csv      forecasting, no shift: MSE, MAE (mean/std over seeds)
  forecast_shift.csv      forecasting, 4 shift conditions + their mean
  classify_clean.csv      classification, no shift: AUROC, AUPRC, accuracy, macro F1/precision/recall
  classify_shift.csv      classification, 4 shift conditions + their mean

To run the notebook: keep it next to the results/ folder. Needs Python 3 with pandas and jupyter.

Conventions
  MSE/MAE lower is better; AUROC/F1 etc. higher is better (classification metrics in percent).
  *_vs_baseline_pct: paired change vs the vanilla baseline on matching seeds, positive = better.
  Shifted = average over the 4 shift conditions (density, timing, variables, composed).
  TMix is excluded. PAM is 8-class, so it has no AUROC/AUPRC.
  P12/P19 seed 0 has no F1/precision/recall (see n_seeds_f1_prec_rec).
