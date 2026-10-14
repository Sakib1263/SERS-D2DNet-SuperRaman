"""Generate the control center and focused stage notebooks from one shared specification."""

from pathlib import Path
from textwrap import dedent
import subprocess
import sys
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
SETUP = """
import numpy as np
from sersd2d_superraman import __version__
from sersd2d_superraman.provenance import read_json
from sersd2d_superraman.data import load_batch
from sersd2d_superraman.notebook import open_workflow, display_table, setup_plotting, training_plot, spectrum_plot, confusion_plot
workflow = open_workflow(CONFIG)
PROJECT, settings = workflow.project, workflow.config
MODE = settings["mode"]
RUN_TRAINING = MODE == "demo" or RUN_RESEARCH_TRAINING
setup_plotting()
display_table(["Control", "Value"], [
    ["Software", __version__], ["Mode", MODE], ["Devices", ", ".join(settings["sources"])],
    ["Translation width / epochs", f"{settings['translation_config']['model']['width']} / {settings['translation_config']['training']['epochs']}"],
    ["Classifier epochs", settings["classifier_config"]["training"]["epochs"]],
    ["Classifier / translation device", f"{settings['classifier_device']} / {settings['translation_device']}"],
    ["Research training requested", RUN_RESEARCH_TRAINING]])
"""


def md(text):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text):
    return nbf.v4.new_code_cell(dedent(text).strip())


def write(name, title, goal, cells):
    book = nbf.v4.new_notebook(
        cells=[
            md(
                f"# {title}\n\n**SERS-D2DNet + SuperRaman · v0.2.0**\n\n[Notebook guide](../docs/NOTEBOOK_GUIDE.md) · [Paper alignment](../docs/PAPER_IMPLEMENTATION_MAP.md) · [README](../README.md)\n\n## Goal\n{goal}"
            ),
            md(
                "## Setup\nSelect the Python environment where `.[notebooks]` is installed. Edit `CONFIG` below to select the demo or your research configuration. Every notebook opens the same persistent run. Configuration changes require a new `run_directory`. The default executes a synthetic, one-epoch, width-2 demonstration across all four portable devices."
            ),
            code('CONFIG = "configs/workflow_demo.json"\nRUN_RESEARCH_TRAINING = False'),
            code(SETUP),
            md(
                "### Read the execution scope\nThe demo verifies software behavior. Its plots and scores are synthetic and have no scientific performance interpretation. Research mode uses authorized preprocessed arrays and the recovered width-128 graph. Published-result reproduction remains unresolved; review the paper alignment document before reporting results."
            ),
            *cells,
            md(
                "## Checks\nArtifact reuse checks file hashes and input/configuration identities. Test rows are evaluated after checkpoint selection; they do not choose the models."
            ),
            code('display_table(["Stage", "Status"], workflow.status().items())'),
            md(
                "## Next Steps\nFor real experiments, copy `configs/workflow_research.json` to `configs/local/my_run.json`, set the data path and a new run directory, and select it in `CONFIG`. Set RUN_RESEARCH_TRAINING = True explicitly after reviewing the displayed settings. See the [notebook guide](../docs/NOTEBOOK_GUIDE.md) for stage navigation and interpretation."
            ),
        ]
    )
    book.metadata.kernelspec = {
        "display_name": "Python 3 (select SERS environment)",
        "language": "python",
        "name": "python3",
    }
    book.metadata.language_info = {"name": "python"}
    book.metadata.sers_pipeline = {
        "version": "0.2.0",
        "mode": "control_center",
        "default_data": "synthetic",
        "paper_reproduction_certified": False,
    }
    # Stable IDs keep regeneration diffs focused on content.
    for index, cell in enumerate(book.cells):
        cell.id = f"{Path(name).stem}-{index:02d}"
    book.cells[3].metadata["jupyter"] = {"source_hidden": True}
    nbf.validate(book)
    nbf.write(book, ROOT / "notebooks" / name)


PREP = [
    md(
        "## Steps\n### 1. Prepare and audit the input roles\nThe interface accepts preprocessed `[N,1400]` spectra in `[0,1]`. Rows retain device, analyte, sample identity, optional Raman axis, and optional acquisition group. Class-matched training pairs are physically unpaired. Split checks keep duplicates and acquisition groups together."
    ),
    code(
        'counts = workflow.prepare()\ndisplay_table(["Device", "Fit", "Validation", "Test"], [[d, r["train"], r["validation"], r["test"]] for d, r in counts.items()])'
    ),
]
TRANSLATE = [
    md(
        "### 2. Train one translator for each portable device\nEvery translator learns a source-to-Renishaw mapping independently. Deep supervision uses named heads; inference selects only the 1400-bin `out` head. The recovered graph is retained. A maximum of 50 class-matched fit pairs is requested in research mode; actual per-class counts are recorded, since small independent fit splits can contain fewer."
    ),
    code(
        'translation_runs = workflow.train_translations() if RUN_TRAINING else {}\nif not RUN_TRAINING:\n    print("Research translation training is disabled. Set RUN_RESEARCH_TRAINING = True in Setup to run it.")\ndisplay_table(["Device", "Best epoch", "Validation MAE", "Fit pairs/class (min–max)"], [[d, r["best_epoch"], round(r["best_validation_out_mae"], 5), str(min(r["fit_class_counts"].values())) + "–" + str(max(r["fit_class_counts"].values()))] for d, r in translation_runs.items()])'
    ),
]
CLASSIFY = [
    md(
        "### 3. Train the reusable Renishaw classifier\nSuperRaman learns from the Renishaw fit role and selects checkpoints on independent Renishaw validation. The maintained implementation uses cross-entropy on logits and legacy fractional shifts. These differ from the paper's MAE/integer-shift wording and are identified in every configuration and the alignment map."
    ),
    code(
        'classifier_run = workflow.train_classifier() if RUN_TRAINING else None\nif not RUN_TRAINING:\n    print("Research classifier training is disabled. Set RUN_RESEARCH_TRAINING = True in Setup to run it.")\nif classifier_run:\n    print({key: classifier_run[key] for key in ("best_epoch", "best_validation_loss", "objective", "synthetic")})'
    ),
]
EVALUATE = [
    md(
        "### 4. Connect the models and evaluate the held-out cascade\nA bundle copies one classifier and the selected translators. In separate workers, the same test rows follow two paths: original → classifier, and source → translator `out` → per-row `[-1,1]` classifier transform → classifier. Class-mean spectral agreement uses independent Renishaw test spectra."
    ),
    code(
        'ready = (workflow.output / "classification/checkpoint/metadata.json").is_file() and all((workflow.output / "translation" / d / "checkpoint/metadata.json").is_file() for d in settings["sources"])\nreports = {}\nif ready:\n    workflow.build_bundle()\n    reports = workflow.evaluate()\n    display_table(["Device", "Original accuracy", "Translated accuracy", "MAE before", "MAE after"], [[d, round(r["classification"]["original"]["accuracy"], 4), round(r["classification"]["translated"]["accuracy"], 4), round(r["translation"]["before"]["mae"], 5), round(r["translation"]["after"]["mae"], 5)] for d, r in reports.items()])\nelse:\n    print("Evaluation pending: complete both training stages before building the bundle.")'
    ),
    md(
        "### 5. Inspect a class and the classification comparison\nChange `DEVICE` and `CLASS_ID` to inspect the same held-out cohort. Shading shows between-spectrum standard deviation, not a confidence interval. A physical Raman axis is shown only when supplied; otherwise the axis is bin index."
    ),
    code(
        'DEVICE = settings["sources"][0]\nCLASS_ID = 0\nif reports:\n    spectrum_plot(workflow, DEVICE, CLASS_ID)\n    import matplotlib.pyplot as plt\n    plt.show()\n    confusion_plot(workflow, DEVICE)\n    plt.show()'
    ),
]


def main():
    (ROOT / "notebooks").mkdir(exist_ok=True)
    write(
        "00_control_center.ipynb",
        "Pipeline control center",
        "Operate the full two-stage workflow from one notebook: select settings, prepare data, train the independent models, build a portable bundle, and compare predictions.",
        PREP
        + TRANSLATE
        + CLASSIFY
        + EVALUATE
        + [
            md(
                "### 6. Review validation histories\nEach curve shows the criterion used for its model. A one-epoch demo is an execution check, not evidence of convergence."
            ),
            code("if ready:\n    training_plot(workflow)\n    plt.show()"),
            md(
                "### Paper alignment\nThe flow follows Sections 3.1 and 3.6: separate source-to-Renishaw translators and one reusable Renishaw classifier, compared before and after translation. Exact model/training/result parity is not established. The recovered translator is 61,511,562 parameters versus the paper's 6.15 M; pooling/interpolation and classifier shifts/loss also differ. Independent test isolation is a maintained protocol change."
            ),
        ],
    )
    write(
        "01_data_contract.ipynb",
        "Data preparation and inspection",
        "Prepare and inspect the shared run's data without launching model training.",
        PREP
        + [
            md("### Inspect a bounded batch and its metadata"),
            code(
                'batch = load_batch(workflow.prepared / "Enwave/test.npz")\n{"shape": batch.spectra.shape, "analytes": len(set(batch.labels)), "physical_axis_available": batch.axis_cm1 is not None, "acquisition_groups_available": batch.groups is not None, "synthetic": batch.metadata.get("synthetic", False)}'
            ),
            md("### Inspect the declared label registry"),
            code("from sersd2d_superraman.data.registry import CLASSES\nlist(enumerate(CLASSES))"),
        ],
    )
    write(
        "02_train_translation.ipynb",
        "SERS-D2DNet translation training",
        "Control source-to-Renishaw fitting for every selected portable device, with actual fit counts and checkpoint selection visible.",
        PREP
        + TRANSLATE
        + [
            md("### Check saved full-resolution output contracts"),
            code(
                'if translation_runs:\n    print({d: {"parameters": read_json(workflow.output / "translation" / d / "checkpoint/metadata.json")["parameters"], "outputs": read_json(workflow.output / "translation" / d / "checkpoint/metadata.json")["output_names"]} for d in settings["sources"]})'
            ),
        ],
    )
    write(
        "03_train_classifier.ipynb",
        "SuperRaman classifier training",
        "Control the reusable Renishaw model independently of translator fitting.",
        PREP
        + CLASSIFY
        + [
            md("### Inspect the trained classifier contract"),
            code(
                'checkpoint = workflow.output / "classification/checkpoint/metadata.json"\nif checkpoint.is_file():\n    metadata = read_json(checkpoint)\n    print({k: metadata[k] for k in ("parameters", "architecture_id", "input_transform", "paper_reproduction_certified")})'
            ),
        ],
    )
    write(
        "04_translate_and_classify.ipynb",
        "Cascade evaluation and comparison",
        "Connect completed checkpoints, preserve input identities, and inspect before/after predictions.",
        PREP + EVALUATE,
    )
    write(
        "05_explain_predictions.ipynb",
        "Prediction explanations",
        "Explain a trained run's classifier on explicitly selected original or translated test rows.",
        [
            md(
                "## Steps\n### 1. Select a completed model and input path\nRun the control center first. Gradients below are with respect to classifier inputs scaled to `[-1,1]`, not the raw instrument signal. Select `logit` or `probability` and a fixed target class; SmoothGrad keeps that class fixed across perturbations."
            ),
            code(
                'import torch\nfrom sersd2d_superraman.classification.model import load_checkpoint\nfrom sersd2d_superraman.explainability import explain\nDEVICE = settings["sources"][0]\nUSE_TRANSLATED = True\nTARGET_CLASS = 0\nTARGET = "logit"\nSMOOTH_SAMPLES = 4\nNOISE_STD = 0.01\ncheckpoint = workflow.output / "classification/checkpoint"\nif not (checkpoint / "metadata.json").is_file():\n    raise ValueError("Run 00_control_center or 03_train_classifier before explaining a model.")\ninput_file = workflow.output / "evaluation" / DEVICE / "translated.npz" if USE_TRANSLATED else workflow.prepared / DEVICE / "test.npz"\nbatch = load_batch(input_file, normalized=not USE_TRANSLATED)\nrow = int(np.flatnonzero(batch.labels == TARGET_CLASS)[0])\nmodel, metadata = load_checkpoint(checkpoint, settings["classifier_device"])\ngradient, classes = explain(model, batch.spectra[row:row+1], target_classes=np.array([TARGET_CLASS]), target=TARGET, smooth_samples=SMOOTH_SAMPLES, noise_std=NOISE_STD, seed=37)\nassert gradient.shape == (1,1400) and np.isfinite(gradient).all()'
            ),
            md(
                "### 2. Display the spectrum and explanation\nA gradient indicates sensitivity of this selected score to the input; it does not establish a biochemical mechanism. Demo explanations are synthetic. SHAP and historical BCE-based saliency are outside this release."
            ),
            code(
                'import matplotlib.pyplot as plt\nfrom sersd2d_superraman.data.registry import CLASSES\nx = np.arange(1400) if batch.axis_cm1 is None else batch.axis_cm1\nfig, axes = plt.subplots(2,1,figsize=(10,5),sharex=True,layout="constrained")\naxes[0].plot(x,batch.spectra[row],color="#176B87")\naxes[0].set(title=f"{MODE.title()} / {DEVICE} / {CLASSES[TARGET_CLASS]} / {TARGET} target",ylabel="Input intensity")\naxes[1].plot(x,gradient[0],color="#9B3B54")\naxes[1].set(ylabel="Mean input gradient",xlabel="Spectral bin index" if batch.axis_cm1 is None else "Raman shift (cm⁻¹)")\nplt.show()'
            ),
            md("### 3. Record explanation provenance"),
            code(
                'from sersd2d_superraman.provenance import write_json\nexplanation_directory = workflow.output / "explanations"\nexplanation_directory.mkdir(exist_ok=True)\nfig.savefig(explanation_directory / "saliency.png",dpi=160)\nwrite_json(explanation_directory / "saliency.json", {"mode": MODE, "device": DEVICE, "translated": USE_TRANSLATED, "sample_id": str(batch.sample_ids[row]), "target_class": TARGET_CLASS, "target": TARGET, "smooth_samples": SMOOTH_SAMPLES, "noise_std": NOISE_STD, "seed": 37, "checkpoint_sha256": metadata["weights_sha256"], "gradient_coordinate": "classifier_scaled_input", "paper_reproduction_certified": False})\n{"finite_gradient": bool(np.isfinite(gradient).all()), "target_class": int(classes[0])}'
            ),
        ],
    )
    subprocess.run([sys.executable, "-m", "ruff", "format", str(ROOT / "notebooks")], check=True)
    print("Built six shared-workflow notebooks")


if __name__ == "__main__":
    main()
