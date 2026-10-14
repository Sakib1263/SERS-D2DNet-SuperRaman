# Reproducibility and checkpoint migration

Every maintained run starts in a new output directory. Configuration, input array/label/sample-ID/axis/acquisition-group hashes, pairing manifests, seeds, selected epoch, selection direction, backend versions, and weights hashes are stored. Test spectra do not enter selection. The code does not claim exact continuation of a historical optimizer or random-number state.

The root package and data commands do not import TensorFlow or PyTorch. Each backend loads only for its own operation. `predict` uses separate Python worker processes for translation and classification, transferring data through NPZ. Set up both dependencies in the runner's environment; a classifier-only environment can use `classify` directly.

## Artifacts

A translation checkpoint contains `metadata.json` and the HDF5 weight file named by its `weights` field (ending in `.weights.h5`). Metadata specifies graph source hash, width/depth, the configured output names, full-resolution head, source/target devices, linear output policy, parameters, versions, class registry hash, and training provenance. A classifier checkpoint contains a tensor-only `.pt` state dictionary named by metadata and `metadata.json`, including q, filter width, shift profile, scaling contract, and class order identity. Reads verify hashes and load states strictly.

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

The shared `Workflow` controller records immutable configuration snapshots, per-stage input signatures, output hashes, running/completed/failed states, software version, and package-source hash. It verifies the prepared split manifests and reconstructs original spectral row order before comparing source content hashes. Reopening a completed stage reuses verified artifacts; failed or changed stages require a new run. This is not optimizer resume. All six notebooks share this control surface.

## Compatibility in version 0.2

Canonical batches and compatible model weight states remain readable. Batch identities now bind Raman axes and acquisition groups as well as spectra, IDs, and labels. Older prediction artifacts without those bindings should be regenerated before evaluation. Workflow artifact reuse checks the software version and configuration/input signatures; it is not optimizer continuation. The public synthetic translator fixture checks the focused graph against the preserved original without requiring obsolete source trees.

## Failure handling and compatibility in version 0.3

NPZ outputs are serialized to temporary files on the destination filesystem, then published without overwriting a competing writer. Mutable JSON manifests/histories are replaced atomically. Local NTFS/ext4 hard links are used for no-clobber NPZ publication; unsupported filesystems fail without publishing the target. These contracts cover ordinary process exceptions and cancellation, not a certification of physical power-loss durability.

Checkpoint saves first serialize weights, derive a SHA-256 filename, and publish metadata last. A serialization/metadata exception leaves the previous committed checkpoint readable. After a successful save the previous managed weight file is retired to bound disk use. A hard process kill can leave an unreferenced temporary or weight file; it cannot make a metadata/weight mismatch pass verification. Checkpoint directories support one writer and should not serve concurrent readers during training: create a copied bundle for inference. Existing version-0.2 fixed-name checkpoints remain readable. Consumers must follow `metadata["weights"]` rather than assuming a filename.

Workflow initialization and stages share an exclusive lock. Live configuration snapshots are checked again before each stage, and reuse binds the current package source as well as version/configuration/inputs/outputs. Source edits during a worker invalidate its stage. A version-0.2 workflow should be preserved as historical output; choose a fresh run directory for version 0.3. This still does not resume optimizer state.

Workers write full stdout/stderr to disk, and errors show a bounded tail with a log path. Interrupting a notebook stage marks it failed and sends termination to its POSIX process group, including cascade workers. Native Windows direct-worker cancellation is implemented, but Windows process-tree cancellation is not certified. A kernel termination or machine interruption can leave a stale lock; preserve the old run and select a new directory.

Translation evaluation now verifies the producing input-file hash and the source/target/output policy, plus label/axis/group identities. Old translations without that provenance must be regenerated. Source/reference comparisons remain physically unpaired class-mean comparisons.

The CPU constraints file pins only the two historically verified framework versions. It is not a complete environment lock and does not specify CUDA libraries. `python -m sersd2d_superraman doctor` identifies the current interpreter and installed package metadata without importing the frameworks. This pass did not repeat environment installation, model parity, notebook execution, or hardware tests.
