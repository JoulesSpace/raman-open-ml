# Contributing

Thanks for your interest in improving raman-open-ml.

## Setup

```bash
pip install -e ".[dev,tuning,xai]"     # editable install + optional extras
pre-commit install                      # ruff lint/format + folderinfo on commit
python scripts/download_data.py         # fetch the open datasets into ./data
```

With the editable install you can `import raman_ml` directly (the runner scripts
also work without it via a small `sys.path` shim).

## Before opening a PR

```bash
ruff check src scripts tests            # lint (CI enforces this)
ruff format src scripts tests           # format
bash scripts/folderinfo.sh              # every directory needs a .folderinfo
pytest -q                               # fast unit suite (no downloads / GPU)
```

## Conventions

- **No em-dashes** anywhere (prose, comments, docstrings) - use a spaced hyphen.
- **Every directory carries a one-line `.folderinfo`** (CI lints this).
- **Report what you verified**, not what you assume: numbers in docs must come
  from running the scripts; flag caveats in `agent-memory/notes/honest-limitations.md`.
- New algorithms go in `src/raman_ml/` with a unit test in `tests/test_core.py`;
  cite the source paper in the docstring.
- Keep `src/raman_ml/` dependency-light (numpy/scipy/sklearn/torch); heavier
  tools (optuna, shap, umap) are optional extras imported lazily.
- Conventional commit messages (`feat:`, `fix:`, `docs:`, ...).

## Where things live

See [`CLAUDE.md`](CLAUDE.md) for the repository layout and operating rules, and
[`agent-memory/MEMORY.md`](agent-memory/MEMORY.md) for the design record (ADRs,
insights, the research synthesis, and the roadmap).
