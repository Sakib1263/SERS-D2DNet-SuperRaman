"""Deterministic class-stratified split with duplicate/acquisition groups kept intact."""

import hashlib
import numpy as np
from ..provenance import json_hash


def fingerprints(x):
    return np.asarray(
        [hashlib.sha256(np.ascontiguousarray(row, dtype=np.float32).tobytes()).hexdigest() for row in x]
    )


def independent_split(batch, seed=37, validation_fraction=0.1, test_fraction=0.2):
    batch.validate()
    if batch.labels is None:
        raise ValueError("Split construction requires analyte labels")
    if not (
        0 < validation_fraction < 1 and 0 < test_fraction < 1 and validation_fraction + test_fraction < 1
    ):
        raise ValueError("Validation and test fractions must be positive and sum to less than one")
    n = len(batch.spectra)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return int(i)

    def join(keys):
        seen = {}
        for i, key in enumerate(keys):
            if key in seen:
                parent[find(i)] = find(seen[key])
            else:
                seen[key] = i

    fp = fingerprints(batch.spectra)
    join(fp)
    if batch.groups is not None:
        join(batch.groups)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    # Whole acquisition groups may include multiple analytes. Allocate using class count deficits.
    labels = np.unique(batch.labels)
    counts = np.asarray([np.sum(batch.labels == label) for label in labels])
    if np.any(counts < 3):
        raise ValueError("Every analyte needs at least three independent rows/groups for three split roles")
    validation_target = np.clip(np.floor(validation_fraction * counts), 1, counts - 2)
    test_target = np.clip(np.floor(test_fraction * counts), 1, counts - validation_target - 1)
    targets = np.asarray([counts - validation_target - test_target, validation_target, test_target])
    group_rows = list(groups.values())
    group_counts = [
        np.asarray([np.sum(batch.labels[rows] == label) for label in labels]) for rows in group_rows
    ]
    if np.any(np.sum(np.asarray(group_counts) > 0, axis=0) < 3):
        raise ValueError("Every analyte needs at least three independent groups for three split roles")
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(64):
        buckets = [[], [], []]
        assigned = np.zeros_like(targets)
        order = rng.permutation(len(group_rows))
        # Larger groups first, random order breaks ties reproducibly.
        order = sorted(order, key=lambda j: -len(group_rows[j]))
        for j in order:
            rows = group_rows[j]
            gc = group_counts[j]
            scores = []
            for k in range(3):
                proposed = assigned.copy()
                proposed[k] += gc
                coverage_penalty = float(10 * n * n * np.sum(proposed == 0))
                scores.append(
                    coverage_penalty + float(np.sum(((proposed - targets) / np.maximum(targets, 1)) ** 2))
                )
            k = int(np.argmin(scores))
            assigned[k] += gc
            buckets[k].extend(rows)
        if np.all(assigned > 0):
            best = buckets
            break
    if best is None:
        raise ValueError(
            "Cannot give every analyte independent train/validation/test groups; supply more groups"
        )
    parts = {
        name: np.sort(np.asarray(rows, dtype=np.int64))
        for name, rows in zip(("train", "validation", "test"), best)
    }
    assert_disjoint(*(batch.subset(parts[k]) for k in parts))
    manifest = {
        "schema_version": 1,
        "seed": seed,
        "requested_fractions": {"validation": validation_fraction, "test": test_fraction},
        "input": batch.identity(),
        "grouping": "transitive union of float32 duplicate fingerprints and supplied acquisition IDs",
        "biological_independence": "not established"
        if batch.groups is None
        else "limited by supplied group metadata",
        "parts": {
            k: {
                "rows": v.tolist(),
                "sample_ids": batch.sample_ids[v].tolist(),
                "class_counts": {str(int(c)): int(np.sum(batch.labels[v] == c)) for c in labels},
            }
            for k, v in parts.items()
        },
    }
    manifest["manifest_sha256"] = json_hash(manifest)
    return parts, manifest


def assert_disjoint(*batches):
    for i, left in enumerate(batches):
        for right in batches[i + 1 :]:
            if left.groups is not None and right.groups is not None and set(left.groups) & set(right.groups):
                raise ValueError("Acquisition group overlap between split roles")
            if left.source_device != right.source_device:
                continue
            if set(left.sample_ids) & set(right.sample_ids):
                raise ValueError("Sample identity overlap between split roles")
            if set(fingerprints(left.spectra)) & set(fingerprints(right.spectra)):
                raise ValueError("Duplicate spectrum overlap between split roles")


def pseudo_pairs(source, reference, samples_per_class=50, seed=37):
    """Independently drawn class-matched rows; no physical pair identity is asserted."""
    if source.labels is None or reference.labels is None:
        raise ValueError("Pseudo-pair construction requires labels in both domains")
    if samples_per_class < 1:
        raise ValueError("samples_per_class must be positive")
    if reference.source_device != "Renishaw" or source.source_device == "Renishaw":
        raise ValueError("Translation fits a portable source to the Renishaw reference")
    if set(source.labels) != set(reference.labels):
        raise ValueError("Pseudo-pair domains must contain the same fitting classes")
    rng = np.random.default_rng(seed)
    sr, rr = [], []
    for c in sorted(set(source.labels)):
        a, b = np.flatnonzero(source.labels == c), np.flatnonzero(reference.labels == c)
        size = min(samples_per_class, len(a), len(b))
        sr.extend(rng.choice(a, size, replace=False).tolist())
        rr.extend(rng.choice(b, size, replace=False).tolist())
    src, ref = source.subset(sr), reference.subset(rr)
    manifest = {
        "pairing": "class_matched_physically_unpaired",
        "seed": seed,
        "maximum_per_class": samples_per_class,
        "source": src.identity(),
        "reference": ref.identity(),
        "source_ids": src.sample_ids.tolist(),
        "reference_ids": ref.sample_ids.tolist(),
    }
    manifest["sha256"] = json_hash(manifest)
    return src, ref, manifest
