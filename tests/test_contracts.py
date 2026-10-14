import json
import numpy as np
import pytest
from sersd2d_superraman.data import SpectrumBatch, save_batch, load_batch
from sersd2d_superraman.data.splits import independent_split, assert_disjoint, pseudo_pairs
from sersd2d_superraman.preprocessing import classifier_input
from sersd2d_superraman.evaluation import classification_metrics, translation_metrics
from sersd2d_superraman.pipeline import verify_bundle


def batch(device="Renishaw", per_class=9, seed=7):
    rng = np.random.default_rng(seed)
    n = 20 * per_class
    x = rng.uniform(0, 1, (n, 1400)).astype(np.float32)
    return SpectrumBatch(
        x, device, np.asarray([f"{device}:{i}" for i in range(n)]), np.repeat(np.arange(20), per_class)
    ).validate()


@pytest.mark.parametrize("error", ["shape", "constant", "nan", "range", "labels", "ids", "axis"])
def test_reject_invalid_contract(error):
    b = batch()
    if error == "shape":
        b.spectra = b.spectra[:, :1399]
    if error == "constant":
        b.spectra[0] = 1
    if error == "nan":
        b.spectra[0, 0] = np.nan
    if error == "range":
        b.spectra[0, 0] = 2
    if error == "labels":
        b.labels = b.labels.astype(float)
        b.labels[0] = 0.5
    if error == "ids":
        b.sample_ids[1] = b.sample_ids[0]
    if error == "axis":
        b.axis_cm1 = np.zeros(1400)
    with pytest.raises(ValueError):
        b.validate()


def test_pickle_free_roundtrip_and_explicit_scaling(tmp_path):
    b = batch()
    save_batch(tmp_path / "batch.npz", b)
    loaded = load_batch(tmp_path / "batch.npz")
    assert loaded.identity() == b.identity()
    values = classifier_input(b.spectra)
    assert values.shape == (180, 1, 1400)
    np.testing.assert_allclose(values.min(2), -1, atol=1e-6)
    np.testing.assert_allclose(values.max(2), 1, atol=1e-6)


def test_duplicate_and_acquisition_group_split_is_deterministic():
    b = batch(per_class=12)
    b.spectra[1] = b.spectra[0]
    b.groups = np.asarray([f"group:{i // 2}" for i in range(len(b.spectra))])
    parts, manifest = independent_split(b)
    again, m2 = independent_split(b)
    assert manifest == m2
    assert set(np.concatenate(list(parts.values()))) == set(range(len(b.spectra)))
    for key in parts:
        np.testing.assert_array_equal(parts[key], again[key])
        assert set(b.labels[parts[key]]) == set(range(20))
    assert any(0 in rows and 1 in rows for rows in parts.values())
    assert_disjoint(*(b.subset(rows) for rows in parts.values()))


def test_overlap_rejected_even_when_ids_are_changed():
    b = batch()
    a = b.subset([0, 1])
    c = b.subset([0, 2])
    c.sample_ids = np.asarray(["other1", "other2"])
    with pytest.raises(ValueError, match="Duplicate"):
        assert_disjoint(a, c)


def test_pseudo_pairs_preserve_class_and_pair_provenance():
    source, target = batch("Enwave", seed=2), batch(seed=3)
    x, y, manifest = pseudo_pairs(source, target, samples_per_class=4)
    assert len(x.spectra) == 80
    np.testing.assert_array_equal(x.labels, y.labels)
    assert manifest["pairing"] == "class_matched_physically_unpaired"
    x2, y2, m2 = pseudo_pairs(source, target, samples_per_class=4)
    np.testing.assert_array_equal(x2.spectra, x.spectra)
    np.testing.assert_array_equal(y2.spectra, y.spectra)
    assert manifest == m2


def test_classification_metrics_against_confusion_counts():
    y = np.array([0, 0, 1, 1, 1])
    logits = np.zeros((5, 20))
    logits[np.arange(5), [0, 1, 1, 1, 0]] = 10
    metrics = classification_metrics(y, logits)
    assert metrics["accuracy"] == pytest.approx(3 / 5)
    assert metrics["weighted_recall"] == pytest.approx(3 / 5)
    assert metrics["confusion_matrix"][0][:2] == [1, 1]
    assert metrics["weighted_f1"] == pytest.approx(0.6)
    assert metrics["samples"] == 5


def test_translation_is_class_mean_agreement_and_keeps_negative_r2():
    source = batch("Enwave")
    reference = batch(seed=99)
    result = translation_metrics(source, source.spectra, reference)
    assert result["scope"].startswith("class_mean")
    assert result["per_class"][0]["before"] == result["per_class"][0]["after"]
    from sersd2d_superraman.evaluation.metrics import _agreement

    assert _agreement(np.array([3, 4]), np.array([0, 1]))["r2"] < 0
    assert "Calculate_DEOI" in result["dsc_definition"]
    assert _agreement(np.array([0.0, 1.0]), np.array([1.0, 1.0]))["dsc"] == pytest.approx(2 / 3)


def test_bundle_refuses_path_escape(tmp_path):
    from sersd2d_superraman.data.registry import registry_hash

    obj = {
        "schema_version": 1,
        "class_registry_sha256": registry_hash(),
        "classifier": {"directory": "../escape", "metadata_sha256": "bad"},
        "translators": {},
    }
    (tmp_path / "bundle.json").write_text(json.dumps(obj))
    with pytest.raises(ValueError, match="within"):
        verify_bundle(tmp_path / "bundle.json")


def test_historical_config_cannot_silently_run_as_maintained(tmp_path):
    from sersd2d_superraman.cli import _config

    path = tmp_path / "historical.json"
    path.write_text(json.dumps({"executable": False}))
    with pytest.raises(ValueError, match="not executable"):
        _config(path)


def test_batches_are_not_overwritten(tmp_path):
    path = tmp_path / "batch.npz"
    save_batch(path, batch())
    with pytest.raises(ValueError, match="overwrite"):
        save_batch(path, batch())
