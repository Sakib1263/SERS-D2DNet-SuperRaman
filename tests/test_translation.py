import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
from sersd2d_superraman.translation.model import (
    TranslationConfig,
    build_model,
    primary_output,
    save_checkpoint,
    load_checkpoint,
)
from sersd2d_superraman.translation.training import (
    supervision_targets,
    train_translation,
    TranslationTrainingConfig,
)
from sersd2d_superraman.data.splits import independent_split
from test_contracts import batch


def test_actual_supplied_graph_contract():
    model = build_model()
    assert model.count_params() == 61511562
    assert model.output_names == ["level1", "level2", "level3", "out"]
    assert [tuple(x.shape[1:]) for x in model.outputs] == [(700, 1), (350, 1), (175, 1), (1400, 1)]
    del model
    tf.keras.backend.clear_session()


def test_supervision_pooling_means():
    a = np.arange(1400, dtype=np.float32)[None, :]
    targets = supervision_targets(a)
    assert targets["level3"].shape == (1, 175, 1)
    assert targets["level3"][0, 0, 0] == 3.5
    np.testing.assert_array_equal(targets["out"][:, :, 0], a)


def test_translation_backward_save_load_and_batch_parity(tmp_path):
    config = TranslationConfig(width=2)
    model = build_model(config)
    x = batch("Enwave").spectra[:2]
    with tf.GradientTape() as tape:
        values = model([x[:, :, None]], training=True)
        loss = sum(tf.reduce_mean(tf.abs(v)) for v in values)
    gradients = tape.gradient(loss, model.trainable_weights)
    assert all(g is not None and bool(tf.reduce_all(tf.math.is_finite(g))) for g in gradients)
    before = primary_output(model, x, batch_size=1)
    save_checkpoint(tmp_path / "ckpt", model, config, "Enwave")
    loaded, _, _ = load_checkpoint(tmp_path / "ckpt")
    np.testing.assert_allclose(before, primary_output(loaded, x, batch_size=2), rtol=1e-5, atol=1e-6)


def test_tiny_translation_training_and_selection(tmp_path):
    source, ref = batch("Enwave", per_class=5, seed=2), batch(per_class=5, seed=3)
    a, _ = independent_split(source)
    b, _ = independent_split(ref)
    _, report = train_translation(
        source.subset(a["train"]),
        ref.subset(b["train"]),
        source.subset(a["validation"]),
        ref.subset(b["validation"]),
        tmp_path / "fit",
        TranslationConfig(width=2),
        TranslationTrainingConfig(epochs=1, batch_size=32, samples_per_class=1),
    )
    assert report["test_used_for_selection"] is False
    assert report["best_epoch"] == 1
    assert report["pairing"] == "class_matched_physically_unpaired"


def test_benchmark_averages_repetitions_once_and_records_device():
    from sersd2d_superraman.evaluation.benchmark import benchmark

    model = build_model(TranslationConfig(width=2))
    report = benchmark(model, batch("Enwave").spectra[:1], "translation", warmup=1, repetitions=2)
    assert report["mean_batch_ms"] == pytest.approx(np.mean(report["individual_batch_ms"]))
    assert report["requested_execution_device"] == "cpu"
    if not tf.config.list_physical_devices("GPU"):
        with pytest.raises(ValueError, match="GPU requested"):
            benchmark(model, batch("Enwave").spectra[:1], "translation", device="cuda", repetitions=1)


def test_focused_graph_matches_original_weight_and_output_fixture():
    from pathlib import Path

    path = Path(__file__).parent / "fixtures/translator_reference.npz"
    with np.load(path, allow_pickle=False) as fixture:
        model = build_model(TranslationConfig(width=2))
        weights = [fixture[f"weight:{i}"] for i in range(len(model.weights))]
        model.set_weights(weights)
        outputs = model([fixture["input"]], training=False)
        for name, output in zip(model.output_names, outputs):
            np.testing.assert_allclose(output.numpy(), fixture[f"output:{name}"], rtol=1e-6, atol=1e-7)
