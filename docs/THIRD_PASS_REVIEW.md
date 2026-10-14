# Version 0.3 reliability and usability review

15 October 2026. Reviewed from failure handling, cancellation, editable-source reuse, efficiency, first-run usability, packaging hygiene, bibliography, and editorial consistency. Numerical model behavior and the scientific protocol were preserved. Existing numerical, notebook, hardware, real-data, and installed-wheel tests were not repeated.

## Changes

| Finding | Resolution |
|---|---|
| Direct NPZ writes could leave a partial final artifact | Same-filesystem temporary serialization and atomic no-clobber publication |
| A failed checkpoint update could invalidate the last good model | Content-addressed weights with metadata published last; rollback preserves the previous committed pair |
| Cancellation at the metadata-publication boundary could remove newly referenced bytes | Commit-aware cleanup preserves the committed metadata/weight pair |
| Long jobs captured their full worker output in memory | File-backed combined logs and bounded error tails; one progress line per training epoch |
| Cancelling the controller could leave nested cascade workers running | POSIX process-group ownership is inherited by nested workers; a real grandchild cancellation scenario passed |
| Interrupted stages remained marked as running | Interrupted workers are marked failed; locks are released during ordinary cancellation |
| Concurrent or incomplete initialization could expose partial workflow state | Initialization shares the stage lock and publishes workflow state after configuration snapshots |
| Cached stages could reuse results after editable code changed without a version bump | Package-source identity contributes to each stage signature and is checked after worker completion |
| A live notebook could use edited configuration snapshots | Snapshots are checked again before each stage |
| Translation evaluation matched rows without confirming the producing input | Producing file hash, domain/output policy, and label/axis/group identities are checked |
| Split checks repeatedly constructed the same hashes and group sets | Each role is hashed once per disjointness call; cross-device acquisition comparisons use one union per role |
| New users lacked a concise environment diagnostic | Backend-free `doctor` command, CPU framework constraints, clone/install instructions, and troubleshooting table |
| Historical outputs and release artifacts cluttered the checkout | Clean notebook sources, fresh private validation output directories, and external recovery of old generated content |
| Bibliography and prose needed consistency | Full associated-paper credits, method-role bibliography, corrected source-hash wording, duplicate-row removal, and historical/current evidence labels |

## New verification only

Twenty-five new checks passed in two non-overlapping runs: 22 initial cases and three additional cases. They exercise publication failures and races, cancellation and a real nested process, workflow initialization, live snapshot changes, source-aware caching, translation provenance, strict JSON, bundle declarations, diagnostics, and hash-work efficiency. Framework writers are represented by bytes for publication fault injection; no model is trained by these checks. The new `doctor` CLI was also inspected in the existing WSL interpreter.

Changed Python files received static checks. Public documentation links and notebook source changes were reviewed without executing notebook cells. Detailed counts and scope are recorded in [THIRD_PASS_VERIFICATION.json](THIRD_PASS_VERIFICATION.json). The [version-0.2 record](VERIFICATION.json) remains historical; it is not a fresh claim for this release.

## Compatibility and use

Read checkpoint filenames from `metadata.json["weights"]`. Existing fixed-name checkpoints remain readable. Version-0.3 workflow signatures require a fresh run directory when moving from version 0.2. Completed stages in the current version reuse artifacts only when source, configuration, inputs, and outputs still match. This is not optimizer resume.

Use [00_control_center.ipynb](../notebooks/00_control_center.ipynb) as the primary interface. Start with the synthetic profile, then create a local research configuration with authorized preprocessed data. Full-width research training and independent scientific evaluation remain separate work. The [paper alignment map](PAPER_IMPLEMENTATION_MAP.md) preserves unresolved paper/source differences.

## Cleanup and practical limits

Old notebook previews and generated demo models, installed-wheel verification outputs, old test reports/caches, the obsolete installation-check environment, and version-0.2 release binaries were moved to the sibling recovery directory `SERS-D2DNet-SuperRaman-Recovery-20261015-ThirdPass`. Active research data and the development environment were retained. Useful source-compatibility fixtures and regression tests remain; `test_stage_contracts.py` replaces the historical pass-based test filename. One-time editing utilities are kept outside the checkout.

No prior model/parity/notebook/hardware/data/installation test was rerun. Fresh installation, updated framework serialization, full notebook execution, a version-0.3 binary release, native Windows process-tree cancellation, and physical power-loss durability were not certified by this review. GitHub Actions retains the complete regressions for future pushes. No commit, push, or publication was performed.
