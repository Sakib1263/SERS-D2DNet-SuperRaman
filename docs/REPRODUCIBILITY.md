# Reproducibility and checkpoint migration

Every maintained run starts in a new output directory. Configuration, input array/label/sample-ID/axis/acquisition-group hashes, pairing manifests, seeds, selected epoch, selection direction, backend versions, and weights hashes are stored. Test spectra do not enter selection. The code does not claim exact continuation of a historical optimizer or random-number state.

The root package and data commands do not import TensorFlow or PyTorch. Each backend loads only for its own operation. `predict` uses separate Python worker processes for translation and classification, transferring data through NPZ. Set up both dependencies in the runner's environment; a classifier-only environment can use `classify` directly.

## Artifacts

A translation checkpoint contains `model.weights.h5` and `metadata.json`. Metadata specifies graph source hash, width/depth, four output names, full-resolution head, source/target devices, linear output policy, parameters, versions, class registry hash, and training provenance. A classifier checkpoint contains a tensor-only `model.pt` state dictionary and `metadata.json`, including q, filter width, shift profile, scaling contract, and class order identity. Reads verify hashes and load states strictly.

`bundle` copies these artifacts into a self-contained directory with relative paths and a manifest. It requires all four portable translators unless `--allow-partial` is explicit. An untrained initialization requires `--allow-untrained`; demonstration status is retained. Bundles do not certify research accuracy.

## Historical classifier objects

PyTorch full-object checkpoints depend on their original class import paths and may execute pickle code. Use only your trusted original files and a separate environment. The conversion command makes this trust decision explicit:

```bash
sers-pipeline convert-classifier --input /trusted/paper_model.pt \
  --legacy-source /path/to/original/classifier_code \
  --config configs/maintained/classifier.json \
  --parity-input /path/to/representative_preprocessed.npz \
  --trust-legacy-pickle --output output/converted_classifier
```

For a 64-filter historical XAI model, provide a configuration with `model.filters=64`; do not load it as the 16-filter model. Full-object migration requires logits to match on the supplied parity batch before saving. A state-only historical checkpoint has no original executable model for comparison; its report marks parity as unchecked. Keep original files and their hashes.

Translation `.keras` or old HDF5 files need graph and layer-shape verification against the recovered source before they can be adopted. No such trained file was supplied. The maintained weight save/load path is tested, but a historical translator conversion is not certified in this release.

## Determinism and hardware

Split/pseudo-pair sampling is deterministic under the recorded seed. Torch and TensorFlow seeds are set for initialization and training. CUDA `grid_sample` backward can be nondeterministic; the code records that limitation instead of promising bitwise GPU reproducibility. Compare states/outputs within declared tolerances.

Verified local framework versions are TensorFlow 2.21.0, PyTorch 2.6.0+cu126, NumPy 2.5.1, and h5py 3.14.0 under Python 3.13.14. CPU model checks cover both stages. Hardware checks and measurements are recorded separately; TensorFlow GPU support was unavailable in the inspected runtime, even though PyTorch sees CUDA.

For isolated verification of an existing environment:

```bash
python -m venv --system-site-packages .venv_verify
# Linux/WSL: use the new environment's Python explicitly.
.venv_verify/bin/python -m pip install --no-deps --no-build-isolation -e .
```

On Windows, the corresponding virtual-environment executable is `.venv_verify/Scripts/python.exe`. The verified local environment for this work is WSL/Python 3.13.14.

The exact local evidence and remaining reproduction work are summarized in [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md). The original notebooks' validation/test reuse, training schedule differences, metrics, and parameter-count discrepancy are historical findings, not erased by the new implementation.

## Notebook run state

The shared `Workflow` controller records immutable configuration snapshots, per-stage input signatures, output hashes, running/completed/failed states, and software version. It verifies the prepared split manifests and reconstructs original spectral row order before comparing source content hashes. Reopening a completed stage reuses verified artifacts; failed or changed stages require a new run. This is not optimizer resume. All six notebooks share this control surface.

## Compatibility in version 0.2

Canonical batches and compatible model weight states remain readable. Batch identities now bind Raman axes and acquisition groups as well as spectra, IDs, and labels. Older prediction artifacts without those bindings should be regenerated before evaluation. Workflow artifact reuse checks the software version and configuration/input signatures; it is not optimizer continuation. The public synthetic translator fixture checks the focused graph against the preserved original without requiring obsolete source trees.
