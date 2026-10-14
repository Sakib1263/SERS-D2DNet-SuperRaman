"""Optional input-gradient explanation with explicit fixed targets."""


def explain(model, spectra, target_classes=None, target="logit", smooth_samples=1, noise_std=0.0, seed=37):
    import numpy as np
    import torch
    from ..preprocessing import classifier_input

    if (
        target not in ("logit", "probability")
        or type(smooth_samples) is not int
        or smooth_samples < 1
        or not np.isfinite(noise_std)
        or noise_std < 0
    ):
        raise ValueError("Invalid saliency target/noise settings")
    values = classifier_input(spectra)
    device = next(model.parameters()).device
    model.eval()
    with torch.inference_mode():
        logits = model(torch.from_numpy(values).to(device))
        classes = logits.argmax(1).cpu().numpy() if target_classes is None else np.asarray(target_classes)
    if (
        classes.shape != (len(values),)
        or not np.issubdtype(classes.dtype, np.number)
        or np.iscomplexobj(classes)
        or not np.isfinite(classes).all()
        or np.any(classes != np.floor(classes))
        or np.any((classes < 0) | (classes > 19))
    ):
        raise ValueError("target_classes must contain one valid class ID per row")
    classes = classes.astype(np.int64)
    rng = np.random.default_rng(seed)
    gradients = np.zeros_like(values)
    for _ in range(smooth_samples):
        noise = rng.normal(0, noise_std, values.shape).astype(np.float32)
        x = torch.from_numpy(values + noise).to(device).requires_grad_(True)
        scores = model(x)
        if target == "probability":
            scores = torch.softmax(scores, dim=1)
        selected = scores[torch.arange(len(x), device=device), torch.as_tensor(classes, device=device)]
        gradients += torch.autograd.grad(selected.sum(), x)[0].detach().cpu().numpy()
    return gradients[:, 0, :] / smooth_samples, classes
