# Contributing and release preparation

Use a dedicated Python 3.11–3.13 environment and the existing notebook controller. Keep numerical model changes separate from workflow/documentation changes. Before changing an operator, document its tensor, scaling, sampling, and checkpoint contracts and add a meaningful regression. Preserve the focused translator's source bytes and architecture identity; do not reformat that file.

## Development checks

Install `.[notebooks,dev]`. For a future complete validation run:

```bash
python -m ruff check src tests tools
python -m pytest -q
python tools/validate_notebooks.py
```

Use `python -m pytest -q tests/test_reliability.py` when reviewing only the failure/publication/controller cases. New checks should cover behavior that matters to users; retain the classifier and translator fixtures because they protect source compatibility. Do not commit datasets, checkpoints, virtual environments, generated notebook previews, or local configuration paths. File a bug with the command/notebook, software and Python versions, configuration stripped of private paths, and the relevant worker-log tail.

`tools/build_notebooks.py` is the notebook specification. It generates clean notebooks with no execution outputs. `tools/validate_notebooks.py` executes a new synthetic run and writes executed copies/previews under a fresh ignored directory; public notebook sources stay clean. Never use research data in public examples or validation artifacts.

## Prepare a release

Update the package version, `CITATION.cff`, changelog, notebook specification, and current evidence together. Preserve historical verification records with their actual version/date; do not present them as fresh checks.

```bash
python -m build
python -m twine check --strict dist/*.whl dist/*.tar.gz
python tools/release_contents.py
```

The release helper rejects ambiguous wheel/source candidates and checks source content before writing the source ZIP and manifest. Use a clean `dist/` for each release. A Git push publishes source changes; release binaries are separate artifacts and must be rebuilt for the changed version. GitHub Actions repeats the maintained regression checks on future pushes. Local checks do not establish remote CI success, full scientific training, or reproduction of paper results.
