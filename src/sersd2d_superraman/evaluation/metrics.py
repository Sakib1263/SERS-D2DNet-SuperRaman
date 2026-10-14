"""Inspectably defined metrics; unpaired translation is evaluated on class means."""

import numpy as np
from ..data.registry import CLASSES


def classification_metrics(labels, logits):
    y = np.asarray(labels)
    scores = np.asarray(logits)
    if y.ndim != 1 or scores.shape != (len(y), 20) or len(y) == 0:
        raise ValueError("Expected [N] labels and [N,20] logits")
    if (
        not np.isfinite(scores).all()
        or not np.isfinite(y).all()
        or np.any(y != np.floor(y))
        or np.any((y < 0) | (y > 19))
    ):
        raise ValueError("Invalid labels or logits")
    pred = scores.argmax(axis=1)
    cm = np.zeros((20, 20), dtype=np.int64)
    np.add.at(cm, (y.astype(int), pred), 1)
    tp = cm.diagonal().astype(float)
    support = cm.sum(axis=1).astype(float)
    fp, fn = cm.sum(axis=0) - tp, support - tp
    tn = len(y) - tp - fp - fn

    def divide(a, b):
        return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)

    precision, recall = divide(tp, tp + fp), divide(tp, tp + fn)
    f1, specificity = divide(2 * tp, 2 * tp + fp + fn), divide(tn, tn + fp)
    weights = support / support.sum()
    return {
        "accuracy": float(tp.sum() / len(y)),
        "weighted_precision": float(weights @ precision),
        "weighted_recall": float(weights @ recall),
        "weighted_f1": float(weights @ f1),
        "weighted_specificity": float(weights @ specificity),
        "samples": len(y),
        "zero_denominator_policy": "zero; classes with zero support have zero aggregation weight",
        "confusion_matrix": cm.tolist(),
        "class_order": list(CLASSES),
        "per_class": [
            {
                "id": i,
                "name": CLASSES[i],
                "support": int(support[i]),
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "specificity": float(specificity[i]),
            }
            for i in range(20)
        ],
    }


def _agreement(estimate, target):
    a, b = np.asarray(estimate, dtype=float), np.asarray(target, dtype=float)
    diff = a - b
    variance = float(np.sum((b - b.mean()) ** 2))
    norms = float(np.linalg.norm(a) * np.linalg.norm(b))
    stds = float(np.linalg.norm(a - a.mean()) * np.linalg.norm(b - b.mean()))
    # Recovered Calculate_DEOI: continuous overlap after removing negative intensity.
    # Work on copies; the historical ravel-based routine modified predictions in place.
    nonnegative_a, nonnegative_b = np.maximum(a, 0), np.maximum(b, 0)
    dice_denom = float(nonnegative_a.sum() + nonnegative_b.sum())
    return {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "pcc": None if stds == 0 else float((a - a.mean()) @ (b - b.mean()) / stds),
        "r2": None if variance == 0 else float(1 - np.sum(diff**2) / variance),
        "cosine_similarity": None if norms == 0 else float(a @ b / norms),
        "dsc": None
        if dice_denom == 0
        else float(2 * np.minimum(nonnegative_a, nonnegative_b).sum() / dice_denom),
    }


def translation_metrics(source, translated, reference):
    if reference.source_device != "Renishaw":
        raise ValueError("Translation evaluation requires a Renishaw reference")
    if source.labels is None or reference.labels is None:
        raise ValueError("Class-mean evaluation requires source and reference labels")
    a = np.asarray(translated)
    if a.shape != source.spectra.shape or not np.isfinite(a).all():
        raise ValueError("Translated spectra must match source rows and be finite")
    rows = []
    for c in sorted(set(source.labels)):
        si, ri = source.labels == c, reference.labels == c
        if not ri.any():
            raise ValueError(f"Reference evaluation contains no class {c}")
        target = reference.spectra[ri].mean(axis=0)
        rows.append(
            {
                "class_id": int(c),
                "source_count": int(si.sum()),
                "reference_count": int(ri.sum()),
                "before": _agreement(source.spectra[si].mean(axis=0), target),
                "after": _agreement(a[si].mean(axis=0), target),
            }
        )
    summary = {}
    for role in ("before", "after"):
        summary[role] = {}
        for key in rows[0][role]:
            vals = [r[role][key] for r in rows if r[role][key] is not None]
            summary[role][key] = None if not vals else float(np.mean(vals))
            summary[role][key + "_defined_classes"] = len(vals)
    return {
        "scope": "class_mean_agreement; physically unpaired domains",
        "per_class": rows,
        "macro_class_mean": summary,
        "dsc_definition": "2 sum(min(max(a,0),max(b,0)))/(sum(max(a,0))+sum(max(b,0))); recovered Calculate_DEOI, without input mutation",
        "r2_definition": "1 - SSE/SS_reference; original notebook reversed sklearn r2_score arguments",
        "overall_aggregation": "macro mean of class metrics; original notebook also computed metrics on a grand-mean spectrum",
    }
