# OPERA manuscript

**Compile `sigconf.tex`.** It is the only driver: it holds the preamble and front matter, and pulls in each
section with `\input`. In Overleaf, set `sigconf.tex` as the main document. Locally:

```bash
latexmk -pdf sigconf.tex
```

No LaTeX toolchain is installed in this repo, so the manuscript has not been built here.

## Layout

```
sigconf.tex              driver: preamble, macros, title/authors, abstract wrapper, \input list, bibliography
sections/abstract.tex    abstract body only (the abstract environment is in the driver)
sections/01_introduction.tex   problem, phenomenon, the privileged-observation insight, 3 contributions; Figure 2
sections/02_related_work.tex   three distinctions: encoders, missingness mechanisms, privileged information
sections/03_problem.tex        formalisation: X, S, O = S(X); process family; re-observation protocol; constraints
sections/04_method.tex         POSD as the centrepiece, then augmentation, then optional mining; Algorithm 1, Figure 1
sections/05_setup.tex          datasets, observation-process splits, metrics, protocol
sections/06_results_shift.tex  Results head; 6.1 does the shift matter, 6.2 does augmentation help; Tables 2-4
sections/07_results_method.tex 6.3 POSD, 6.4 mining, 6.5 unseen families, 6.6 ablations and cost; Tables 5-7
sections/08_discussion.tex     the two mechanisms' different jobs, the ratio pitfall, limitations
sections/09_conclusion.tex
refs.bib                 26 references, all cited, none unused
figures/*.pdf            generated from results; see below
acmart.cls, ACM-Reference-Format.bst
```

A section file contains **only** its `\section{...}` and body. It must not declare a preamble, load a package
or define a macro; shared macros (`\Sobs`, `\Oobs`, `\prelim`, `\todo`) live in the driver.

## Do not add amsmath or amssymb

`acmart` already provides `booktabs`, `graphicx` and `xcolor`, and under XeLaTeX/LuaLaTeX it also loads
`amssymb` + `unicode-math`. Requesting `amssymb` here raises

```
LaTeX Error: Command \Bbbk already defined.
```

The sections therefore avoid every amsmath-only construct (`align`, `\boxed`, `\text{}` in math), so the
document compiles unchanged under pdfLaTeX, XeLaTeX and LuaLaTeX. If you need `amsmath` for new material,
select pdfLaTeX and add `amsmath` only, never `amssymb`.

## Figures

All five figures are generated from the results rather than drawn by hand, and all five are referenced in the
text (`fig:framework`, `fig:views`, `fig:curve`, `fig:unseen`, `fig:ablation`). Regenerate after new runs:

```bash
python scripts/generate_figures.py --dataset physionet   # writes ../figures
cp ../figures/*.pdf figures/
```

## Conventions

* `\prelim` (dagger) marks every number obtained from a single seed. The protocol requires five paired seeds.
* Contribution hierarchy (see `CLAUDE_paper.md`): privileged observation self-distillation is the central
  mechanism, observation-process augmentation is the supporting one, hard mining is a conditional extension.
  Do not restore the earlier framing in which the three were co-equal.
* Classification degradation is computed on `1 - AUROC` (or `1 - accuracy`), never on cross-entropy.
* `\todo{...}` marks a pending measurement. Do not delete either marker without replacing the number.
* Numbers come from `results/raw/*/results.json`, aggregated into
  `results/aggregated/tables_<dataset>.md` by `scripts/aggregate.py`.

## Venue metadata

The driver compiles with `[sigconf,nonacm]` and a suppressed reference block so it builds standalone. For a
real ACM submission, drop `nonacm`, uncomment the `\setcopyright` / `\acmConference` / `\acmISBN` block, and
paste the CCS XML generated at <https://dl.acm.org/ccs> where indicated.
