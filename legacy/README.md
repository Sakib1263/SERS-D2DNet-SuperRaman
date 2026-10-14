# Preserved sources

The supplied `SERSD2DNet/` and `SuperRamanNet/` trees remain unchanged in the local checkout. Their initial file inventory and SHA-256 values are in `docs/audit/source_inventory_2026-10-14.json`. A local `source_snapshot.zip` preserves this inventory as a recoverable archive and is excluded from the public payload.

The public package contains the necessary author-supplied translator graph and a focused classifier implementation. It does not include private datasets, stale result images, compiled caches, broad inherited baseline utilities, or executable legacy experiments with test-set selection. Historical settings are recorded separately under `configs/legacy/`.
