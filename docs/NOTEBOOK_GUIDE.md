# Notebook operating guide

## Start here

Open [00_control_center.ipynb](../notebooks/00_control_center.ipynb) and select a Python kernel containing the installed package and its `notebooks` extra. The controls live in the Setup cell and the chosen JSON configuration. Every stage notebook reads that same file and opens the same `workflow.json`; stage outputs are shared.

The checked-in notebook outputs show a synthetic demonstration. Rerunning uses the controls you select. Run the cells from top to bottom; do not edit computed artifacts manually.

## Controls

| Control | Purpose |
|---|---|
| `CONFIG` | Demo configuration or your local research configuration |
| `run_directory` | Dedicated output directory; change it for a new experiment |
| `mode` | `demo` generates synthetic inputs; `research` requires authorized data |
| `sources` | One or more unique portable devices; all four is the complete workflow |
| `dataset` / `prepared_data` | Exactly one research input: HDF5/NPZ source or prepared study |
| `classifier_config` | Architecture, cross-entropy training, validation selection, seed |
| `translation_config` | Width/depth, weighted MAE, sampling, validation selection, seed |
| `classifier_device` | PyTorch CPU or an available CUDA device |
| `translation_device` | TensorFlow `cpu` or `cuda`; defaults to CPU independently |
| `RUN_RESEARCH_TRAINING` | Explicit research-training switch in training notebooks |

Demo controls use width 2, one epoch, and one fit pair per class to keep execution bounded. Research defaults use the recovered width-128 translator and a maximum of 50 class-matched fit pairs. These profiles carry distinct architecture IDs and no paper-reproduction certification.

## Run the demonstration

Leave `CONFIG` at `configs/workflow_demo.json` and run the control center. It prepares all five devices, trains four translators and one classifier, copies a complete bundle, evaluates all four held-out portable batches, and renders comparisons. The first execution performs work; later executions verify and reuse artifacts. Open notebooks 01–05 for focused controls and inspection of that run.

## Run research

1. Copy `configs/workflow_research.json` into ignored `configs/local/my_run.json`.
2. Set `prepared_data` to an existing prepared study, or replace it with `dataset` pointing to an authorized HDF5 file or canonical five-device NPZ directory.
3. Select a fresh output directory and the available devices. Review all architecture/training settings against the paper alignment map.
4. Point every notebook's `CONFIG` to this file. Run Setup and data preparation first.
5. Set `RUN_RESEARCH_TRAINING = True` in the control center to launch the full training stages. The focused notebooks can launch each stage separately.
6. Build/evaluate after both independent model stages have finished. Inspect histories, fit counts, comparison metrics, and sample identities before making scientific claims.

Full research training is a new experiment. It is not restoration of historical optimizer state or reproduction of a certified paper checkpoint. Research mode rejects synthetic inputs and never replaces missing files with demonstration data.

## Read the figures

- Validation histories show the criterion used for checkpoint selection, separate from test results.
- Spectral overlays show independent class means and ±1 standard deviation. They are not physically paired sample comparisons or confidence intervals.
- Confusion matrices compare identical source test rows before and after transformation, in fixed analyte order.
- Saliency/SmoothGrad explains a selected logit or softmax target with respect to the scaled classifier input. Class targets and noise settings are explicit; gradients are not biological evidence.
- Physical Raman coordinates appear only when a verified axis is supplied. Otherwise plots use spectral bin index.

## Reopen and troubleshoot

Reopen the same run through `Workflow(PROJECT, settings)`. A completed stage is reused only when its inputs, settings, and recorded output hashes match. Configuration snapshots are stored in the run directory. Edit the source configuration and choose a new run directory rather than modifying those snapshots.

A failed or interrupted stage is recorded as such; its directory is retained. Inspect the reported worker error and start a new run after fixing the cause. The controller does not silently skip failed work or mix partly completed experiments.

If a framework is missing, install the notebook extra into the selected kernel environment. If TensorFlow cannot access CUDA, use `translation_device="cpu"`; this does not prevent a working PyTorch CUDA classifier. If `CONFIG` cannot be found, check that the notebook is opened within the repository. The Python API and CLI use the same model/artifact contracts as the notebook frontend.

## Maintainer regeneration

`tools/build_notebooks.py` is the sole notebook source generator. Regenerate only when changing the specification, then run `tools/validate_notebooks.py`. Validation executes all six notebooks with one shared synthetic run, writes HTML/PNG previews under ignored `tmp/second_pass_notebooks/`, and retains the verified synthetic outputs in the notebooks. A research configuration must not be used for public notebook regeneration.
