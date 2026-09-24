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
sections/01_introduction.tex
sections/02_related_work.tex
sections/03_problem.tex        formalisation: X, S, O = S(X); process family; descriptor; constraints
sections/04_method.tex         the three mechanisms, the objective, Algorithm 1, deployment
sections/05_setup.tex          datasets, observation-process splits, metrics, protocol
sections/06_phenomenon.tex     Tables 2-3: the phenomenon and the matched-density comparison
sections/07_results.tex        Tables 4-6: mechanism isolation, cross-setting, cost
sections/08_analysis.tex       negative results, the degradation-ratio pitfall, limitations
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

Figures are generated from the results rather than drawn by hand:

```bash
python scripts/generate_figures.py --dataset physionet   # writes ../figures
cp ../figures/*.pdf figures/
```

## Conventions

* `\prelim` (dagger) marks every number obtained from a single seed. The protocol requires five.
* `\todo{...}` marks a pending measurement. Do not delete either marker without replacing the number.
* Numbers come from `results/raw/*/results.json`, aggregated into
  `results/aggregated/tables_<dataset>.md` by `scripts/aggregate.py`.

## Venue metadata

The driver compiles with `[sigconf,nonacm]` and a suppressed reference block so it builds standalone. For a
real ACM submission, drop `nonacm`, uncomment the `\setcopyright` / `\acmConference` / `\acmISBN` block, and
paste the CCS XML generated at <https://dl.acm.org/ccs> where indicated.
