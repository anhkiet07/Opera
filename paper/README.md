# OPERA manuscript

`main.tex` (ACM sigconf, `nonacm`) + `refs.bib` + `figures/`. No LaTeX toolchain is installed in this repo;
upload the folder to Overleaf or run `latexmk -pdf main.tex` where one is available. `acmart.cls` and
`ACM-Reference-Format.bst` are included.

Figures are generated from the results, not hand-drawn:

```bash
python scripts/generate_figures.py --dataset physionet   # writes ../figures, copy into paper/figures
```

Conventions in the source:

* `\prelim` (dagger) marks every number obtained from a single seed. The protocol requires five.
* `\todo{...}` marks a measurement that is pending. Do not delete either marker without replacing the
  underlying number.
* Numbers come from `results/raw/*/results.json` via `results/aggregated/tables_<dataset>.md`.
