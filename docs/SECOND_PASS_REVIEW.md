# Comprehensive second-pass review

Version 0.2.0, 15 October 2026. This pass reviews the active primary workflow against the paper, recovered sources, model contracts, notebook behavior, and release boundaries. It improves scientific transparency and software usability; it does not certify published-result reproduction.

## Implementation fixes

| Finding | Change and practical effect |
|---|---|
| Disconnected notebook sessions | Shared `Workflow` controller, immutable configuration snapshots, persistent stage records, verified reuse |
| No notebook end-to-end entry point | Control center prepares, trains, bundles, evaluates, and plots all four devices |
| TensorFlow placement implicit in cascade | Separate `translation_device` and classifier `device`; unavailable TensorFlow CUDA raises an error |
| Hashed bundles could still contain semantically wrong metadata | Checkpoint role, device, input layout/range, target, output head, registry, status, and parameter validation |
| Data identities omitted axis/acquisition grouping | Both now contribute to batch identities |
| Prepared content could change with IDs preserved | Workflow reconstructs original row order and verifies the spectral hash in the sealed split manifest |
| Classifier validation could omit classes | All 20 labeled classes required in fit and validation |
| Translation validation class/device roles insufficiently checked | All 20 labeled classes and source/reference roles checked before fitting |
| NaN weight decay or explanation settings accepted | Finite optimizer/noise/target checks and integer target indexing |
| Prediction evaluation trusted IDs without input content | Input identity and fixed class-registry identity also verified |
| Checkpoint metadata missed completed-run summary | Best weights retain final training provenance, including epochs completed and run scope |
| Translator sampling cap obscured actual sample counts | Fit counts and whether 50 pairs/class was achieved are recorded explicitly |
| Broad translator source included inactive model builders | Removed three inactive builders; remaining active graph is checked against original weights/outputs |
| Already-translated input could be transformed twice | Reject it in translation/cascade; use classification directly |
| Already-translated input could be transformed twice | Reject it in translation/cascade; use classification directly |
| One-time author-local audit tools in public tree | Retired to external recovery storage; reusable tests/build/validation tools retained |

## Cleanup and preservation

The supplied 145 original files were hash-verified before moving the duplicate source trees outside the checkout. The recovery directory is the sibling `SERS-D2DNet-SuperRaman-Recovery-20261014`, with a manifest and original snapshot. It also retains previous audit reports, old source distributions, the obsolete wheel-testing environment, and one-time source-discovery scripts. Current research splits and the active verification environment remain available.

The active TensorFlow architecture is a focused extraction of the supplied `IMUNet` builder. Its original source hash remains in checkpoint provenance; its current implementation hash is recorded separately. Shape/parameter, original-weights output-parity, save/load, and backward checks govern acceptance. No pooling/interpolation behavior is silently replaced during cleanup.

## Scientific alignment

The sequence and domain roles follow the paper: independently fitted portable-to-Renishaw translators and one reusable Renishaw classifier, evaluated on source and transformed spectra. Exact equivalence cannot be claimed for the recovered implementation. See the [paper implementation map](PAPER_IMPLEMENTATION_MAP.md) and [machine-readable paper contract](paper_contract.json).

The parameter-count difference cannot be resolved by arbitrarily narrowing the recovered graph. The paper's classifier MAE target encoding is unspecified and conflicts with the recovered cross-entropy implementation. Fixed-integer and learned-fractional shifts are different operators. The maintained protocol isolates test data instead of reproducing validation/test reuse and fit/evaluation overlap from historical workflows. Small independent classes can contain fewer than 50 translation fit pairs. These are visible differences requiring new experiments or recovered evidence.

## Verification and remaining work

Current executed checks are recorded in [VERIFICATION.json](VERIFICATION.json). They include contract/regression tests, six shared-workflow notebooks, original-graph parity, all-four-device synthetic cascade, real prepared-data validation, and built-artifact integrity checks.

Full width-128 research fitting, independent efficacy assessment, recovered paper checkpoint parity, raw preprocessing recovery, and edge-device performance remain scientific work. SHAP, paper baselines, chemistry holdouts, joint training, and new operator substitutions are separate extensions. GitHub publication and remote CI status are independent of local checks.
