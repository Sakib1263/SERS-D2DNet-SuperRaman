from pathlib import Path
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from sersd2d_superraman.classification.model import (
    SuperRaman,
    ClassifierConfig,
    PolynomialShift1d,
    save_checkpoint,
    load_checkpoint,
    predict_logits,
)
from sersd2d_superraman.classification.training import train_classifier, ClassifierTrainingConfig
from sersd2d_superraman.explainability import explain
from sersd2d_superraman.pipeline import create_bundle, verify_bundle
from test_contracts import batch


def test_default_parameter_count_and_kernel_branch_shape():
    m = SuperRaman().eval()
    assert sum(x.numel() for x in m.parameters()) == 448572
    x = torch.randn(2, 1, 1400)
    assert [getattr(m, f"conv{k}")(x).shape[-1] for k in (1, 3, 5, 7)] == [1402, 1400, 1398, 1396]
    assert m(x).shape == (2, 20)
    assert torch.isfinite(m(x)).all()


def test_degree_one_zero_shift_matches_conv():
    layer = PolynomialShift1d(2, 3, 5, ClassifierConfig(q=1, max_shift=0))
    x = torch.randn(2, 2, 25)
    expected = torch.nn.functional.conv1d(x, layer.weights, layer.bias, padding=1)
    torch.testing.assert_close(layer(x), expected)


def test_bounded_profile_applies_bounds_and_removes_singleton_height_shift():
    layer = PolynomialShift1d(1, 2, 3, ClassifierConfig(shift_mode="bounded_fractional_1d"))
    with torch.no_grad():
        layer.shifts[:] = torch.tensor([[20.0, 8.0]])
    x = torch.randn(1, 1, 40)
    a = layer(x)
    with torch.no_grad():
        layer.shifts[:] = torch.tensor([[2.0, 0.0]])
    torch.testing.assert_close(a, layer(x))


def test_legacy_reference_forward_and_input_gradients():
    fixture = Path(__file__).parent / "fixtures/legacy_classifier_reference.npz"
    if not fixture.exists():
        pytest.fail("Legacy parity fixture is required in the public source release")
    with np.load(fixture, allow_pickle=False) as values:
        model = SuperRaman().eval()
        model.load_state_dict(
            {key[6:]: torch.from_numpy(values[key]) for key in values.files if key.startswith("state:")}
        )
        x = torch.from_numpy(values["input"]).requires_grad_(True)
        output = model(x)
        np.testing.assert_allclose(output.detach().numpy(), values["logits"], rtol=1e-5, atol=1e-6)
        output[:, 3].sum().backward()
        np.testing.assert_allclose(x.grad.numpy(), values["input_gradients"], rtol=2e-5, atol=2e-6)


def test_checkpoint_batch_parity_hash_and_portable_bundle(tmp_path):
    m = SuperRaman().eval()
    b = batch(per_class=5)
    before = predict_logits(m, b.spectra[:3], batch_size=1)
    save_checkpoint(tmp_path / "ckpt", m)
    loaded, _ = load_checkpoint(tmp_path / "ckpt")
    np.testing.assert_allclose(
        before, predict_logits(loaded, b.spectra[:3], batch_size=3), rtol=1e-5, atol=1e-6
    )
    bundle = create_bundle(
        tmp_path / "bundle", tmp_path / "ckpt", {}, allow_partial=True, allow_untrained=True
    )
    meta, _ = verify_bundle(bundle)
    assert meta["contains_untrained_models"]
    with (tmp_path / "ckpt/model.pt").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="hash"):
        load_checkpoint(tmp_path / "ckpt")


def test_tiny_training_uses_only_validation(tmp_path):
    b = batch(per_class=5)
    from sersd2d_superraman.data.splits import independent_split

    parts, _ = independent_split(b)
    model, report = train_classifier(
        b.subset(parts["train"]),
        b.subset(parts["validation"]),
        tmp_path / "fit",
        training_config=ClassifierTrainingConfig(epochs=1, batch_size=32),
    )
    assert report["test_used_for_selection"] is False
    assert report["best_epoch"] == 1
    assert model.training is False


def test_explanation_targets_and_fixed_smoothgrad():
    model = SuperRaman().eval()
    x = batch().spectra[:2]
    a, classes = explain(model, x, target="logit", smooth_samples=2, noise_std=0.01)
    b, again = explain(model, x, target="logit", smooth_samples=2, noise_std=0.01)
    assert a.shape == (2, 1400) and np.isfinite(a).all()
    np.testing.assert_array_equal(classes, again)
    np.testing.assert_allclose(a, b)
