# SERS-D2DNet and SuperRaman

An installable research pipeline for device-to-device SERS translation and 20-class analyte classification. It contains the supplied TensorFlow SERS-D2DNet graph, a compatible PyTorch SuperRaman implementation, independent validation, portable checkpoints, and a runner connecting the stages in separate processes.

The implementation accompanies Sakib Mahmud et al., **Deep learning-based cross-device standardization of surface-enhanced Raman spectroscopy for enhanced bacterial recognition**, *Spectrochimica Acta Part A* 347 (2026), 126931. [Paper DOI](https://doi.org/10.1016/j.saa.2025.126931).

**Release status: 0.1.0 research beta.** The supplied translator has 61,511,562 parameters, while the paper reports 6.15 million. The primary SuperRaman has 448,572 parameters. Original trained paper checkpoints are not included or available in the inspected experiment folders. This release prepares and validates the software; it does not certify reproduction of the published performance. See [paper implementation map](docs/PAPER_IMPLEMENTATION_MAP.md).

## Installation

Use Python 3.11-3.13. In a fresh environment, from the directory containing `pyproject.toml`:

```bash
python -m pip install ".[all]"
# Or install only the stage you need:
python -m pip install ".[translation]"
python -m pip install ".[classification]"
```

The base package provides data contracts and NumPy metrics without importing either ML backend. TensorFlow is pinned to the verified 2.21.0 stack. PyTorch 2.6 is the verified classifier stack; newer PyTorch versions must pass the contract checks before research use. Install the hardware-appropriate PyTorch build using its official instructions. CUDA availability is checked separately for each backend; one backend seeing a GPU does not establish support for the other.

To preserve an existing research environment, use a separate virtual environment or register the package there with `python -m pip install --no-deps -e .`. The local implementation work used a separate virtual environment inheriting the existing WSL packages; it did not upgrade that environment.

## Working demonstration

```bash
sers-pipeline demo --output output/synthetic_demo
```

This generates synthetic 20-class data, prepares independent splits, trains each stage for one epoch, creates a portable partial Enwave bundle, and runs the cascade. The demo translator has width 2 for bounded runtime; the supplied width-128 model remains the default research architecture. Demo predictions are software checks and do not measure scientific efficacy.

Outputs include `prediction/translated.npz`, original and translated logits/probabilities, and `prediction/report.json`. The report preserves sample identities and records input/model hashes. Use a new output directory for each run.

## Research workflow

The original HDF5 dataset uses `<Device>_GT` and `<Device>_GT_Labels` keys. Data are not redistributed here. Obtain authorized data as described in [DATA.md](docs/DATA.md), then:

```bash
sers-pipeline prepare --dataset /path/to/SERS_D2DNet_Dataset.h5 --output data/prepared
sers-pipeline train-classifier --data data/prepared --config configs/maintained/classifier.json --output output/classifier
sers-pipeline train-translation --data data/prepared --source Enwave --config configs/maintained/translation.json --output output/enwave
```

Repeat translator training for Tec5, Fdefender, and RID. Train the reusable classifier on Renishaw only. The trainers use validation for checkpoint selection and never take test data as a selection input. The preparation step groups exact float32 duplicates and any supplied acquisition IDs. Without biological acquisition IDs, biological independence remains unestablished.

```bash
sers-pipeline bundle --classifier output/classifier/checkpoint \
  --translator Enwave=output/enwave/checkpoint \
  --translator Tec5=output/tec5/checkpoint \
  --translator Fdefender=output/fdefender/checkpoint \
  --translator RID=output/rid/checkpoint --output output/model_bundle

sers-pipeline predict --bundle output/model_bundle/bundle.json \
  --input data/prepared/Enwave/test.npz --output output/enwave_test

sers-pipeline evaluate-translation --input data/prepared/Enwave/test.npz \
  --translated output/enwave_test/translated.npz --reference data/prepared/Renishaw/test.npz \
  --output output/enwave_test/translation_metrics.json
```

Use `--allow-partial` when bundling a selected portable device. Renishaw inputs bypass translation. Prediction processes are separate, so classifier inference does not need to import TensorFlow. Canonical input is float32 `[N,1400]` with source device and sample IDs; labels are optional at inference. Raw-spectrum baseline correction/resampling is deliberately deferred until its original algorithm and Raman grid are recovered.

The legacy classifier's per-spectrum `[-1,1]` transform is explicit and consistent for original and translated inputs. Translation selects the named full-resolution `out` head and retains linear, unclipped outputs. See [reproducibility](docs/REPRODUCIBILITY.md) and [model cards](docs/MODEL_CARDS.md).

## Evaluation, explanations, and timing

```bash
sers-pipeline classify --checkpoint output/classifier/checkpoint --input data/prepared/Renishaw/test.npz --output output/reference_predictions.npz
sers-pipeline evaluate-classification --input data/prepared/Renishaw/test.npz --predictions output/reference_predictions.npz --output output/reference_metrics.json
sers-pipeline explain --checkpoint output/classifier/checkpoint --input data/prepared/Renishaw/test.npz --target logit --output output/saliency.npz
sers-pipeline benchmark --kind classification --checkpoint output/classifier/checkpoint --input data/prepared/Renishaw/test.npz --batch-size 1 --output output/classifier_timing.json
```

Classification reports accuracy, weighted precision/recall/F1/specificity, class counts, and confusion matrices. Translation compares class means from physically unpaired domains. Continuous-overlap Dice follows the recovered original `Calculate_DEOI` formula without mutating predictions; R2 uses the reference spectrum as ground truth, correcting the original notebook's reversed arguments. Explanations distinguish logit/probability targets and optional fixed-target SmoothGrad; physical axis labels require supplied axis metadata. Timing uses warmup, device synchronization, repeated measurements, and one repetition average. [Benchmark protocol](docs/BENCHMARKS.md).

Historical object checkpoints can be migrated with `convert-classifier`, a trusted source directory, explicit configuration, a parity input batch, and `--trust-legacy-pickle`. Modern artifacts contain tensor state dictionaries and JSON metadata. [Migration instructions](docs/REPRODUCIBILITY.md).

## Verification and release

```bash
python -m pip install ".[all,dev]"
python -m pytest -q
python tools/build_notebooks.py
python tools/validate_notebooks.py
python -m build
python -m twine check --strict dist/*
python tools/release_contents.py
```

See [implementation status](docs/IMPLEMENTATION_STATUS.md) for completed checks and remaining scientific work. The original supplied folders remain preserved locally and are excluded from the public release payload. The standalone package has no required `prismsers`, `sersd2d_ablation`, private data path, or neighboring checkout.

## Citation and credits

Use [CITATION.cff](CITATION.cff) for the paper and record this software version, configuration, and checkpoint hashes in experiments. Credit the paper's eight authors and the original ONN, Self-ONN, Super Neuron, UNet3+, and PSP-Net researchers. [NOTICE.md](NOTICE.md) distinguishes method attribution, preserved source, and maintained implementation scope.

Code is provided under MIT. Dataset access, trained weights, and the paper's CC BY 4.0 license have separate scopes. This software models known analyte labels; it is not a validated clinical diagnostic system.
