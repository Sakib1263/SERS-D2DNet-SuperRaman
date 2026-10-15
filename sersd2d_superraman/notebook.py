"""Small notebook presentation helpers; plotting libraries are imported only on demand."""

from pathlib import Path
import numpy as np
from .data import load_batch
from .data.registry import CLASSES
from .provenance import read_json


def setup_plotting():
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.15,
        }
    )


def training_plot(workflow):
    import matplotlib.pyplot as plt

    paths = [("SuperRaman", workflow.output / "classification/history.json", "validation_loss")]
    paths += [
        (d, workflow.output / "translation" / d / "history.json", "validation_out_mae")
        for d in workflow.config["sources"]
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), layout="constrained")
    for name, path, key in paths:
        rows = read_json(path)
        axis = axes[0] if name == "SuperRaman" else axes[1]
        axis.plot([r["epoch"] for r in rows], [r[key] for r in rows], marker="o", label=name)
        axis.set_xlabel("Epoch")
    axes[0].set(title="Renishaw classifier validation", ylabel="Cross-entropy loss")
    axes[1].set(title="Portable-to-Renishaw validation", ylabel="Full-resolution MAE")
    from matplotlib.ticker import MaxNLocator

    for axis in axes:
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        if len(axis.lines[0].get_xdata()) == 1:
            axis.set_xlim(0.5, 1.5)
            axis.set_xticks([1])
        axis.legend()
    fig.suptitle(f"{workflow.config['mode'].title()} run: checkpoint selection uses validation only")
    return fig


def spectrum_plot(workflow, device="Enwave", class_id=0):
    import matplotlib.pyplot as plt

    if class_id not in range(20):
        raise ValueError("class_id must be 0-19")
    source = load_batch(workflow.prepared / device / "test.npz")
    translated = load_batch(workflow.output / "evaluation" / device / "translated.npz", normalized=False)
    reference = load_batch(workflow.prepared / "Renishaw/test.npz")
    fig, axis = plt.subplots(figsize=(10, 4), layout="constrained")
    x = np.arange(1400) if source.axis_cm1 is None else source.axis_cm1
    for name, batch, color, style in (
        (device, source, "#B85C38", "--"),
        ("Translated", translated, "#176B87", "-"),
        ("Renishaw", reference, "#333333", ":"),
    ):
        values = batch.spectra[batch.labels == class_id]
        axis.plot(x, values.mean(0), label=f"{name} (n={len(values)})", color=color, linestyle=style)
        axis.fill_between(
            x, values.mean(0) - values.std(0), values.mean(0) + values.std(0), color=color, alpha=0.12
        )
    axis.set(
        title=f"{workflow.config['mode'].title()} / {CLASSES[class_id]}: mean and ±1 SD",
        xlabel="Spectral bin index" if source.axis_cm1 is None else "Raman shift (cm⁻¹)",
        ylabel="Intensity (translator output is unclipped)",
    )
    axis.legend()
    return fig


def confusion_plot(workflow, device="Enwave"):
    import matplotlib.pyplot as plt

    report = read_json(workflow.output / "evaluation" / device / "report.json")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout="constrained")
    for axis, role in zip(axes, ("original", "translated")):
        metric = report["classification"][role]
        cm = np.asarray(metric["confusion_matrix"])
        image = axis.imshow(cm, cmap="Blues", vmin=0, vmax=max(1, cm.max()))
        axis.set_xticks(range(20), CLASSES, rotation=90, fontsize=8)
        axis.set_yticks(range(20), CLASSES, fontsize=8)
        axis.set(
            title=f"{role.title()} accuracy: {metric['accuracy']:.1%}",
            xlabel="Predicted analyte",
            ylabel="True analyte",
        )
        axis.grid(False)
        fig.colorbar(image, ax=axis, label="Spectrum count", shrink=0.75)
    fig.suptitle(f"{workflow.config['mode'].title()} / {device}: same held-out rows on both paths")
    return fig


def open_workflow(configuration="configs/workflow_demo.json"):
    """Resolve notebook controls relative to the checkout, with explicit validation overrides."""
    import os
    from .workflow import Workflow

    import tomllib

    project = None
    for candidate in (Path.cwd(), *Path.cwd().parents):
        manifest = candidate / "pyproject.toml"
        if (
            manifest.is_file()
            and tomllib.loads(manifest.read_text(encoding="utf-8")).get("project", {}).get("name")
            == "sersd2d-superraman"
        ):
            project = candidate
            break
    if project is None:
        raise ValueError("Open the notebook within the cloned repository")
    configuration = Path(os.environ.get("SERS_WORKFLOW_CONFIG", configuration))
    if not configuration.is_absolute():
        configuration = project / configuration
    settings = read_json(configuration)
    if os.environ.get("SERS_WORKFLOW_OUTPUT"):
        settings["run_directory"] = os.environ["SERS_WORKFLOW_OUTPUT"]
    return Workflow(project, settings)


def display_table(headers, rows):
    """Render bounded native Markdown tables without exposing hidden notebook state."""
    from IPython.display import Markdown, display

    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(map(cell, headers)) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        if len(row) != len(headers):
            raise ValueError("Table row does not match its headers")
        lines.append("| " + " | ".join(map(cell, row)) + " |")
    display(Markdown("\n".join(lines)))
