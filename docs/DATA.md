# Data contract and access

The **[Multi-Instrument SERS Dataset](https://zenodo.org/records/23372983)** is hosted on Zenodo with restricted file access. The public record identifies Sakib Mahmud as the creator, version **1.0.0**, DOI **[10.5281/zenodo.23372983](https://doi.org/10.5281/zenodo.23372983)**, and a CC BY 4.0 license. Access is granted following curator review of a reasonable request. Research files, paper-trained weights, and historical saliency outputs are not bundled with this software. The demonstration data and both model-compatibility fixtures are synthetic.

## Request access and configure local data

1. Open the Zenodo record and sign in. Request access with a brief intended-use statement and your affiliation, if applicable.
2. After approval, download the files. Extract an archive locally before selecting it as a pipeline input.
3. Copy `configs/workflow_research.json` to ignored `configs/local/my_run.json` and set exactly one of `dataset` and `prepared_data`.
4. For a downloaded preprocessed HDF5 file, remove the template's `prepared_data` entry and add `dataset` with the actual local filename. A directory containing all five canonical NPZ files can also be used as `dataset`.
5. Use `prepared_data` only for a study already produced by `prepare`, containing `study.json` and the device-specific split files. Choose a fresh `run_directory`, open the notebook control center, and select your local configuration.

For example, after placing the approved preprocessed HDF5 file under `data/private/`, replace the template's input setting with the following entry, using its actual filename:

```json
"dataset": "data/private/<downloaded-file>.h5"
```

This is a single configuration entry, not a complete configuration file. Do not keep `prepared_data` alongside it. Paths resolve from the repository root; WSL users should use Linux paths. Downloaded data should remain under ignored `data/` or outside the checkout.

The notebook controller prepares independent split roles when `dataset` is selected. To prepare them separately from a terminal:

```bash
sers-pipeline dataset-info
sers-pipeline prepare --dataset "data/private/<downloaded-file>.h5" --output data/prepared
```

After this command succeeds, select `"prepared_data": "data/prepared"` in a fresh research run. An existing `data/prepared` directory must not be overwritten. For canonical NPZ input, the directory must contain `Renishaw.npz`, `Enwave.npz`, `Tec5.npz`, `Fdefender.npz`, and `RID.npz`. File access is handled through Zenodo; the pipeline reads approved local files and requires no embedded access token.

## Dataset citation and file identity

**Mahmud, S. (2026). Multi-Instrument SERS Dataset (Version 1.0.0) [Data set]. Zenodo. [10.5281/zenodo.23372983](https://doi.org/10.5281/zenodo.23372983).** Cite this dataset together with the associated paper. The dataset's record-listed license and access conditions are separate from the software's MIT license; consult the record and supplied files for current terms.

The public metadata was checked on 15 October 2026. Restricted file contents and checksums were not available for inspection. The historical HDF5 identity below describes the author-supplied local file inspected earlier; it does not certify that a downloaded archive or later dataset version has the same bytes. The loader records the actual input-file hash for each preparation.

## Preprocessed HDF5 contract

The inspected original HDF5 has SHA-256 `53c5375fe0e90d19abe50e6c091ee891d4cbfc7edc401080c0f43fbc6407d443`, 29,545 spectra, and the five root-key device arrays described below. Each spectral array has shape `[N,1400]`, float64 storage, finite values, and range [0,1]. Labels are integer IDs 0-19. These properties were recorded during the earlier data audit and were not rechecked for version 0.3. Acquisition, sample, and substrate groups and a physical Raman axis are not stored in that file.

| Device | Spectra key | Label key | Rows |
|---|---|---|---:|
| Renishaw | Renishaw_GT | Renishaw_GT_Labels | 4966 |
| Enwave | Enwave_GT | Enwave_GT_Labels | 4746 |
| Tec5 | Tec5_GT | Tec5_GT_Labels | 6061 |
| Fdefender | Fdefender_GT | Fdefender_GT_Labels | 1692 |
| RID | RID_GT | RID_GT_Labels | 12080 |

The loader performs a declared float64-to-float32 conversion and records the source-file hash. It does not repeat baseline correction or smoothing. The resulting input is already preprocessed, not raw detector intensity.

## Canonical NPZ

Files are written/read without pickle. Required fields: `spectra` float32 `[N,1400]`, `sample_ids` unique strings `[N]`, and `source_device` a scalar string. Optional fields: `labels` integer `[N]`, `axis_cm1` finite increasing `[1400]`, `groups` acquisition-group strings `[N]`, and `metadata_json` a JSON scalar string.

The fixed classes are `LTA, ENT, PCN, Beta-Carotene, 2,5-DHBA, 2,3-DHBA, Vancomycin HCL, PMB, LPS-1, LPS-2, LPS-3, LPS-4, LPS-5, LPS-6, LPS-7, PGN, LPS-8, LPS-9, LPS-10, LPS-11`. All model outputs use this order. Labels are unnecessary for inference; no missing label is filled with zero.

Reject nonfinite values, wrong lengths, duplicate sample IDs, invalid class IDs, malformed axes, and constant signals. Translation inputs must be [0,1]. Translated linear outputs can leave that interval; they are retained unchanged and the classifier applies its explicit per-spectrum scaling.

## Splits and pseudo-pairs

`prepare` assigns whole groups to fitting, validation, or testing using deterministic class-balance targets. Its groups are the transitive union of exact float32 spectrum duplicates and supplied acquisition IDs. It stores original row IDs, group policy, class counts, requested fractions, and content hashes. Rounding/group size can change achieved fractions; the manifest reports actual counts.

Default requested fractions are 70% fitting, 10% validation, and 20% testing. Translation independently samples up to 50 fit spectra per class from each domain and uses separately selected validation pseudo-pairs. A small source class can supply fewer than 50 fitting examples after reserving independent validation/test rows. This is a maintained protocol change from the historical paper setup.

Source/reference pairs share an analyte label but are physically unpaired. Translation evaluation compares independent class means, never treats unrelated source/reference rows as paired ground truth, and preserves source sample IDs in translated outputs.

Exact duplicate grouping is a technical leakage precaution. It cannot establish patient, biological specimen, substrate, session, or laboratory independence without that metadata. Add acquisition IDs to canonical batches if they become available. Optional chemistry holdouts and external datasets require their own separately documented studies.

## Raman axis

Plots default to spectral bin index. The paper's nominal 400-1800 cm^-1 range and 1400 samples do not establish whether one endpoint is excluded. Do not infer an exact 1 cm^-1 axis using an inclusive 1400-point linspace. Raw baseline correction, smoothing boundaries, resampling, and endpoint conventions require original acquisition/preprocessing evidence before they are added.

## Domain identity after translation

`source_device` records the acquisition instrument and remains unchanged after translation. The reserved `metadata.translation` object records the source, Renishaw target domain, checkpoint, and input hashes. Already-translated batches are accepted by `classify` and rejected by `translate` or the full cascade to prevent applying the device mapping twice.
