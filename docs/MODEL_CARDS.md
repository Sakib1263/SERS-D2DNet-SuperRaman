# Model cards

## SERS-D2DNet supplied graph

Task: portable-device spectra to the Renishaw domain. TensorFlow input `[N,1400,1]`, nominal preprocessed [0,1] intensity. Default graph: `MLMRSNet.IMUNet`, depth 3, base width 128, mixed pooling, deep supervision, original source hash recorded. Measured total parameters: 61,511,562. Output selected by name `out`; auxiliary heads have lengths 700, 350, and 175. The final output is linear and not clipped automatically.

Fit: independent same-analyte pseudo-pairs with weighted MAE. Selection: full-resolution validation MAE. Coverage: the known 20-label registry, with separately trained device-specific models. Physical pairs, unseen-device performance, unseen-analyte generalization, and clinical diagnostic efficacy are not established by this release.

The width-2 graph is used only for fast synthetic verification and has a distinct architecture ID. It is not the published/default research model. Exact reproduction of paper counts and results is unresolved; original trained weights were not found.

## SuperRaman compatible classifier

Task: 20 known analyte logits. PyTorch input `[N,1,1400]`, after explicit per-spectrum min-max scaling to [-1,1]. Four branches use kernels 1/3/5/7, q=3, n=16, and shift range initialization +/-2. Branches concatenate along the spectral dimension before average pooling, batch normalization, tanh, flatten, and linear classification. Parameters: 448,572. The n=64 profile represents the wider historical XAI architecture and is a distinct model.

Shift profiles:

- `legacy_fractional`: preserves the historical learned shared two-coordinate interpolation, including the singleton-height coordinate and ineffective legacy bounds. Required for legacy state/output compatibility.
- `bounded_fractional_1d`: clamps the spectral shift to its range and removes singleton-height motion in the forward path. This is an explicitly different operator behavior requiring fresh evaluation.
- `fixed_integer_1d`: fixed rounded initialization, bounded spectral coordinate, no learned shift parameter. This is a separate experiment profile, not a claim that the historical checkpoint used it.

Fit: cross-entropy on logits, reusable classifier trained on Renishaw. Selection: independent Renishaw validation loss, scheduler direction min. Model probabilities are uncalibrated softmax scores. A model prediction is an analyte label; organism associations do not turn it into a clinical diagnosis.

## Combined bundle

A complete bundle contains four translators and one reusable classifier with hashes and class mapping identities. A partial bundle declares supported devices. The runner preserves source sample order, retains translations, and saves both original and translated classifier outputs. No sample labels are needed at inference. Renishaw bypasses translation.

The demonstration bundle is trained on synthetic data for one epoch with a narrow translator. Its accuracy has no biological meaning. A research bundle needs authorized data, independent evaluation, and checkpoint-specific provenance. No pretrained scientific bundle is distributed in 0.1.0.
