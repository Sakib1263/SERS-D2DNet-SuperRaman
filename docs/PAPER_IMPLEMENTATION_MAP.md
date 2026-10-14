# Paper, recovered sources, and maintained implementation

The primary original source is `RAMAN_Task_V2.ipynb` in the author-provided segmentation experiment. Its model-building cells select `MLMRSNet.IMUNet`, depth 3, width 128, length 1400, mixed pooling, and deep supervision. The original `MLMRSNet.py` matches the supplied copy byte for byte. A CPU graph construction measured 61,511,562 parameters, 231 layers, and outputs level1 `[N,700,1]`, level2 `[N,350,1]`, level3 `[N,175,1]`, and out `[N,1400,1]`.

| Item | Paper | Recovered original code | Maintained behavior |
|---|---|---|---|
| Translator parameters | 6.15 M | 61,511,562 measured | Preserve supplied graph and report measured count |
| Semantic levels | Three levels in Fig. 2 | Three encoder iterations plus bottleneck | Preserve source graph; no artificial shrinking |
| Translator training | 300 epochs; plateau patience 15 | 500 epochs; patience 10 in V2 cells 40/44 | Default 300/15 follows reported training schedule; identify as maintained protocol |
| Deep supervision | Described without numeric weights | out 1.0; level1 .9; level2 .8; level3 .7 | Preserve recovered weights and averaged targets |
| Mixed pooling | Explicit half max + half average | max + average followed by learned 1x1 convolution | Preserve supplied graph |
| Interpolation | Bilinear wording | UpSampling1D repeats values | Preserve supplied graph |
| Classifier parameters | 0.45 M | 448,572 measured in original 16-filter model | Preserve state/tensor contract |
| Classifier objective | MAE; TensorFlow scheduling wording | PyTorch cross-entropy | Cross-entropy on logits; discrepancy is explicit |
| Classifier rate/epochs | 3e-5 / 500 | 5e-4 / 1000 | Maintained default 3e-5 / 500; new run identity |
| Classifier validation | Reusable Renishaw classifier intended | Exp. 1 assigns portable test rows to validation | Independent Renishaw validation; tests excluded from fitting/selection |
| Shift mechanism | Random integer shifts in +/-2 | Learnable fractional, two-coordinate sampling; clamp/round results discarded | Legacy compatibility profile plus separately named corrected 1D profiles |
| Input range | [0,1] preprocessing | Classifier rescales each row to [-1,1] | Explicit checkpoint transform used for both comparison paths |
| Branch concatenation | Four branches followed by pooling | Along spectral axis; 16 BN channels; 22,384 flattened features | Preserved |
| XAI model | n=16 primary architecture | XAI duplicate uses n=64 | Unified configurable architecture; checkpoint declares width |
| DSC | Set-overlap equation | `Calculate_DEOI`: continuous minimum intensity overlap after clipping negatives | Same overlap formula without in-place mutation; no early rounding |
| R2 | Reference-ground-truth equation | `r2_score(prediction, reference)` | Correct reference denominator; negative values retained |
| Overall translation metrics | Text describes class aggregation | Also computes metric on an average across class means | Report per-class and macro-class scores, with aggregation explicitly named |
| Classifier timing | 0.0033 ms reported | Timer divides by repetition count twice; no synchronization | Fresh benchmark protocol; no inference about which exact timer generated paper values |

The class-mean metric source is V2 cells 51-53 and 57. `Calculate_DEOI` flattens each spectrum, replaces negative prediction values with zero, and computes twice the overlap divided by total nonnegative intensity. Because the historical function can modify its input array, a new nonmutating implementation avoids changing values used by other metrics. Exact reproduction of its rounded output can be calculated separately from the full-precision maintained results.

No trained research weights were found in either author-provided original experiment folder. There is no checkpoint proof of the configuration that produced Tables 3/5 or the 64-filter XAI results. The source and synthetic compatibility checks support software compatibility, not verification of those published outcomes.

The release accepts already preprocessed inputs. The original notebooks load baseline-corrected, smoothed, normalized arrays and do not recover the original polynomial baseline algorithm or exact 1400-point Raman axis. Raw ingestion remains deferred rather than guessing these settings.

See `configs/legacy/observed_experiment.json` for non-executable historical settings and `configs/maintained/` for the actual maintained defaults. Corrected selection, metric definitions, and optional shift behavior require fresh experiments. The software never stamps a checkpoint as a certified paper reproduction.
