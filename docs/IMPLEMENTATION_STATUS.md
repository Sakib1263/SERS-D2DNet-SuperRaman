# Implementation and release status

14 October 2026, version 0.1.0 research beta. The standalone primary software pipeline is prepared. Full paper-result reproduction and scientific training remain pending. This distinction applies to every checkpoint/bundle and the public README.

## Completed

- Original source discovery in both author-provided SERS-Bacterial experiment folders. The original translator matches the supplied source hash. Its actual graph has 61,511,562 parameters, 231 layers, and the four registered outputs. The primary classifier has 448,572 parameters.
- Installable package with independent backend imports, fixed 20-class registry, explicit preprocessing, pickle-free data exchange, portable configuration, and no required PRISM checkout.
- Independent class-balanced split preparation with duplicate/acquisition groups held intact. All 29,545 real spectra have been prepared locally, keeping research data outside the public payload.
- Standalone translation and classifier trainers, validation-only checkpoint selection, recorded configuration/data hashes, full-resolution output selection, portable weight metadata, and legacy classifier conversion with output parity checks.
- Portable partial/complete bundle handling, separate backend worker processes, original/translated comparison, six explicitly defined translation metrics, classifier metrics/confusion matrices, optional saliency/SmoothGrad, and synchronized repeated benchmarking.
- 28 contract tests covering invalid inputs, grouping/overlap, deterministic pseudo-pairs, metric calculations, source parameter/output contracts, forward/backward behavior, checkpoint/batch parity, artifact hashes, overwrite prevention, historical-config rejection, and benchmark averaging/device handling.
- Synthetic classifier forward/input-gradient fixture generated from the original class. A synthetic original-object conversion produced identical logits. PyTorch CUDA forward/backward and CPU/CUDA output parity passed on the RTX 3080 Ti Laptop GPU; TensorFlow model tests ran on CPU.
- A complete one-epoch synthetic training-to-bundle-to-inference run succeeded. Its translator width is 2; it is a software demonstration. All five notebooks executed, including the actual synthetic cascade, and the saliency plot was visually checked.
- Default width-128 translator inference and classifier input compatibility checked for one real held-out row from each of the 20 analytes on all four portable devices. These initialized/synthetic-trained models establish input compatibility, not efficacy.
- Original supplied files remain unchanged; a local recovery snapshot is retained. The public package excludes legacy clutter, private research data/results, local environment files, and absolute workspace paths.

Machine-readable evidence is in [VERIFICATION.json](VERIFICATION.json). Detailed local audit data and temporary previews remain in excluded directories. GitHub Actions is configured, but has not run remotely.

## Prepared real-data splits

| Device | Fit | Validation | Test | Total |
|---|---:|---:|---:|---:|
| Renishaw | 3482 | 494 | 990 | 4966 |
| Enwave | 3338 | 465 | 943 | 4746 |
| Tec5 | 4259 | 597 | 1205 | 6061 |
| Fdefender | 1196 | 162 | 334 | 1692 |
| RID | 8473 | 1199 | 2408 | 12080 |

These are a new maintained protocol, not the historical paper splits. Each role contains all 20 classes and has no same-device identity/float32-duplicate overlap. No biological independence claim is made without acquisition group metadata. Public releases do not include these research arrays; `prepare` recreates them from authorized input data.

## Remaining scientific work

The original trained translator/classifier weights are absent from the inspected original folders. They are needed to verify which settings produced the published tables and saliency plots. The paper/source differences remain documented: 6.15 M versus 61.5 M translator parameters, classifier loss/schedule, mixed pooling/interpolation, validation/test reuse, R2 conventions, shift behavior, and reported file/timing sizes.

Full maintained research training and independent evaluation must be run before announcing new scientific performance. The provided default training commands use the real prepared splits and produce new run identities; they do not recreate historical optimizer/RNG state. Original raw preprocessing and the precise Raman grid remain unresolved, so raw instrument ingestion is not offered.

Optional paper baselines, LOCO/LOFO chemistry studies, SHAP, new operator/library substitution, cross-framework conversion, and joint training remain separately scoped experiments. The first release contains the primary translation/classification workflow, as recommended in the plan.

## Environment and publication limits

The original WSL research environment was retained. A separate `.venv_verify` inherits its framework packages, with build tools and this package installed only into that verification environment. TensorFlow CUDA libraries were unavailable there; no GPU library or framework upgrade was performed. Keras/NumPy emitted known array-copy deprecation warnings during weight saving; the numerical and checkpoint checks passed.

Release source ZIP, wheel, source distribution, and checksums are prepared locally. No GitHub repository, remote push, release publication, or package-index publication has been performed.
