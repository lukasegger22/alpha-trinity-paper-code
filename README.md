# Alpha Trinity: Regime-Aware Multi-Asset Allocation

A research project testing whether deterministic crisis control improves a
cost-aware MLP/Random-Forest allocation rule. Results are historical, not investment
advice or evidence of a live trading advantage.

## Submission

- Final paper: [PDF](assessment_run/writing/abgegebenes_PAPER.pdf)
- Canonical source: [LaTeX](assessment_run/writing/abgegebenes_PAPER.tex)
- Reproduction details: [protocol](assessment/docs/reproducibility.md)
- Predictor definitions: [features](assessment/docs/feature_documentation.md)
- Evidence: [tables](assessment/outputs/tables), [figures](assessment/outputs/figures)
- Frozen inputs and provenance: [inputs](assessment/inputs)

The canonical manuscript and its bibliography, generated tables and eight figures
are in assessment_run/writing/. All numerical evidence is stored once under
assessment/outputs/. Internal review notes and duplicate exports are not included.

## Reproduce Offline From the Supplied Snapshot

Use Python 3.14 and TeX Live with latexmk and elsarticle.

    make setup
    make test
    make assessment
    make paper

Or run make all after installation. The assessment uses the delivered snapshot,
not a new Yahoo download or saved legacy weights. Numerical manuscript tables
are generated from the CSV evidence and cross-checked before rendering.

The main portfolio period is 2022-01-03 through 2026-05-08. Features at close t
form an order filled at close t+1. The new position first earns the return ending
t+2. Actual trades include market drift and fees reduce NAV at the fill.

## Interpretation

The fixed split favors crisis control on Sharpe and drawdown depth, but not total
return. Annual expanding and rolling portfolio tests do not confirm consistent
improvement. Do not replace these mixed findings with older performance numbers.
The full result and benchmarks are in ablation_study.csv and baseline_comparison.csv.

Original snapshot acquisition time was not recorded. Its checksum and the
September 2026 freeze date are known. Survivorship, historical design selection,
simplified execution and lack of a pristine prospective holdout remain disclosed
limitations. Reproduction is from the exact supplied snapshot, not from an
assumption that a current vendor download reproduces the past.

## Layout

- assessment/src/alpha_trinity_assessment/: canonical research implementation
- assessment/scripts/: thin command-line entry points and paper-table builder
- assessment/config/protocol.json: frozen experimental settings
- assessment/requirements-lock.txt: tested Python dependency versions
- assessment/inputs/: frozen main panel and separately dated SHY supplement
- assessment/outputs/: regenerated numerical evidence and plots
- assessment/docs/: reproduction protocol and feature documentation
- assessment_run/writing/: final paper and supporting generated assets
- tests/: offline unit and frozen-snapshot checks
- src/ai_ls_allocation/: original application/live-trading prototype, not the paper engine

Live-trading automation, credentials and private helper scripts are not included
in the submission branch. No broker orders are required or sent by the research
reproduction commands. The original application source is retained for context;
it is not the canonical reproduction route and its private integrations are omitted.
