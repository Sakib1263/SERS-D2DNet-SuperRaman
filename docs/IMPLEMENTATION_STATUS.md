# Implementation and release status

15 October 2026 · Version 0.2.0 research beta. The comprehensive engineering second pass is complete. Exact agreement with the paper and reproduction of its published results remain unresolved.

## Completed and checked

- Notebook control center plus five focused stage notebooks, sharing one persistent configuration/run and retaining verified synthetic outputs.
- All-four-device synthetic workflow: four independent translators, one Renishaw classifier, portable bundle, and before/after comparisons on 20 held-out rows per device.
- 48 passing contract/regression tests, no failures/errors/skips; static checks pass. The suite includes original classifier forward/input-gradient parity and a public synthetic original-translator weight/output fixture.
- Cleaned translator matches the original on every output head with zero numerical error on the comparison batch. Default graph remains 61,511,562 parameters; classifier remains 448,572.
- All 29,545 prepared real spectra validated against sealed source content and split manifests. Label coverage, acquisition/duplicate isolation, axes/groups, and sample identities are checked.
- PyTorch CUDA forward/backward and CPU/CUDA logits comparison pass. TensorFlow reports no visible GPU and runs on CPU; backend device controls are separate.
- Professional README, notebook operating guide, structured paper contract, reproducibility/model/benchmark documents, and updated release/citation metadata.
- Obsolete trees, audit scripts, inactive model builders, previous artifacts, and the old wheel-testing environment moved outside the checkout. All 145 supplied files remain unchanged and recoverable.

Current evidence is in [VERIFICATION.json](VERIFICATION.json); implementation findings are in [SECOND_PASS_REVIEW.md](SECOND_PASS_REVIEW.md). Source ZIP, wheel, source distribution, and hashes are built locally. GitHub Actions is configured; remote CI success and publication are not implied.

## Prepared real-data roles

| Device | Fit | Validation | Test | Total |
|---|---:|---:|---:|---:|
| Renishaw | 3482 | 494 | 990 | 4966 |
| Enwave | 3338 | 465 | 943 | 4746 |
| Tec5 | 4259 | 597 | 1205 | 6061 |
| Fdefender | 1196 | 162 | 334 | 1692 |
| RID | 8473 | 1199 | 2408 | 12080 |

These are maintained independent roles, not the historical paper protocol. Biological independence cannot be established without specimen/acquisition metadata. Research arrays remain local and excluded from Git/releases.

## Paper alignment and scientific work

The paper/source differences include translator parameter count, mixed pooling, interpolation, classifier objective/shifts, training/selection roles, actual fit counts, metric conventions, and deployment size/timing. They are explicitly mapped in [PAPER_IMPLEMENTATION_MAP.md](PAPER_IMPLEMENTATION_MAP.md) and [paper_contract.json](paper_contract.json).

No paper-trained checkpoint was found in the supplied original experiment folders. Full research fitting and independent evaluation must precede new performance claims. Raw polynomial-baseline parameters, smoothing/resampling boundaries, and the precise Raman grid still require recovery. Optional baselines, chemistry holdouts, SHAP, operator substitutions, cross-framework conversion, and joint training remain separately scoped studies.

Keras/NumPy emits nonfatal array-copy deprecation warnings during weight saving. Numerical parity, gradients, serialization, and integration checks pass. The original WSL environment is preserved; installation/build checks use separate project environments. Current work has not been committed or pushed.
