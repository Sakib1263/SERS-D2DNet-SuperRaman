"""Build focused, safe-default package walkthroughs with nbformat."""

from pathlib import Path
from textwrap import dedent
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
SETUP = """
from pathlib import Path
import os, sys, uuid
import numpy as np
PROJECT = next((p for p in (Path.cwd(), *Path.cwd().parents) if (p / "pyproject.toml").exists()), Path.cwd())
WORK = Path(os.environ.get("SERS_NOTEBOOK_OUTPUT", PROJECT / "output/notebooks")) / uuid.uuid4().hex[:8]
WORK.mkdir(parents=True)
from sersd2d_superraman.data import SpectrumBatch, load_batch, save_batch
from sersd2d_superraman.cli import synthetic, prepare
"""


def md(text):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text):
    return nbf.v4.new_code_cell(dedent(text).strip())


def write(name, title, cells):
    book = nbf.v4.new_notebook(
        cells=[
            md(title),
            md(
                "## Setup\nInstall the package in this kernel's environment. All default data are synthetic; research training is disabled."
            ),
            code(SETUP),
            *cells,
        ]
    )
    book.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    book.metadata.language_info = {"name": "python"}
    book.metadata.sers_pipeline = {"version": "0.1.0", "mode": "tutorial", "default_research_training": False}
    nbf.validate(book)
    nbf.write(book, ROOT / "notebooks" / name)


def main():
    (ROOT / "notebooks").mkdir(exist_ok=True)
    write(
        "01_data_contract.ipynb",
        "# Data contract and independent splits\n\n## Goal\nPrepare inspectable synthetic 20-analyte input with no physical Raman axis.",
        [
            md("### 1. Generate and split"),
            code(
                'synthetic(WORK / "synthetic", per_class=5)\nsummary = prepare(WORK / "synthetic", WORK / "prepared")\nsummary'
            ),
            md("### 2. Inspect the canonical input"),
            code(
                'batch = load_batch(WORK / "prepared/Enwave/test.npz")\n{"shape": batch.spectra.shape, "device": batch.source_device, "labels": sorted(set(batch.labels.tolist())), "metadata": batch.metadata}'
            ),
            md("## Checks"),
            code(
                'assert batch.spectra.shape == (20, 1400)\nassert batch.metadata["synthetic"]\nassert batch.axis_cm1 is None\nassert len(set(batch.sample_ids)) == len(batch.spectra)'
            ),
            md(
                "## Next steps\nUse an authorized HDF5 dataset with `sers-pipeline prepare`. Acquisition independence requires actual group IDs; the demonstration only proves software behavior."
            ),
        ],
    )
    write(
        "02_train_translation.ipynb",
        "# SERS-D2DNet graph and supervised targets\n\n## Goal\nInspect the supplied graph interface using a narrow synthetic verification profile.",
        [
            md("### 1. Build a bounded graph"),
            code(
                'os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")\nfrom sersd2d_superraman.translation.model import build_model, TranslationConfig, primary_output\nfrom sersd2d_superraman.translation.training import supervision_targets\nconfig = TranslationConfig(width=2)\nmodel = build_model(config)\nsynthetic(WORK / "synthetic", per_class=5)\nbatch = load_batch(WORK / "synthetic/Enwave.npz")\n{ "architecture": config.architecture_id, "parameters": model.count_params(), "output_names": model.output_names }'
            ),
            md("### 2. Check the output and supervision"),
            code(
                "translated = primary_output(model, batch.spectra[:2], batch_size=1)\ntargets = supervision_targets(batch.spectra[:2])\nassert translated.shape == (2,1400) and np.isfinite(translated).all()\n{k: value.shape for k, value in targets.items()}"
            ),
            md("### 3. Research training command"),
            code(
                'RUN_RESEARCH_TRAINING = False\ncommand = [sys.executable, "-m", "sersd2d_superraman", "train-translation", "--data", "data/prepared", "--source", "Enwave", "--config", "configs/maintained/translation.json", "--output", "output/enwave"]\nif RUN_RESEARCH_TRAINING:\n    import subprocess\n    subprocess.run(command, cwd=PROJECT, check=True)\nelse:\n    print("Prepared command:", " ".join(command))'
            ),
            md(
                "## Checks and next steps\nThe research default is depth 3/width 128 and 61,511,562 parameters. The narrow graph above is a software check. Recover paper weights and resolve the reported 6.15 M discrepancy before certifying paper reproduction."
            ),
        ],
    )
    write(
        "03_train_classifier.ipynb",
        "# SuperRaman classification contract\n\n## Goal\nInspect logits and explicit input scaling; select research checkpoints on separate Renishaw validation.",
        [
            md("### 1. Inspect the compatible model"),
            code(
                'import torch\nfrom sersd2d_superraman.classification.model import SuperRaman, predict_logits\nfrom sersd2d_superraman.preprocessing import classifier_input\ntorch.manual_seed(37)\nmodel = SuperRaman().eval()\nsynthetic(WORK / "synthetic", per_class=5)\nbatch = load_batch(WORK / "synthetic/Renishaw.npz")\nlogits = predict_logits(model, batch.spectra[:3])\n{"parameters": sum(x.numel() for x in model.parameters()), "input_shape": classifier_input(batch.spectra[:3]).shape, "logit_shape": logits.shape}'
            ),
            md("## Checks"),
            code(
                "assert logits.shape == (3,20) and np.isfinite(logits).all()\nassert sum(x.numel() for x in model.parameters()) == 448572"
            ),
            md("### 2. Research training command"),
            code(
                'RUN_RESEARCH_TRAINING = False\ncommand = [sys.executable, "-m", "sersd2d_superraman", "train-classifier", "--data", "data/prepared", "--config", "configs/maintained/classifier.json", "--output", "output/classifier"]\nif RUN_RESEARCH_TRAINING:\n    import subprocess\n    subprocess.run(command, cwd=PROJECT, check=True)\nelse:\n    print("Prepared command:", " ".join(command))'
            ),
            md(
                "## Next steps\nThis initialized model has no established accuracy. Use the documented split preparation, then evaluate held-out data once selection finishes. Choose a separate shift profile explicitly when changing legacy behavior."
            ),
        ],
    )
    write(
        "04_translate_and_classify.ipynb",
        "# Cascaded inference and comparison\n\n## Goal\nRun a portable bundle in separate backend workers while preserving input IDs.",
        [
            md("### 1. Select artifacts"),
            code(
                'from sersd2d_superraman.pipeline import run_pipeline\nBUNDLE = Path(os.environ.get("SERS_NOTEBOOK_BUNDLE", PROJECT / "output/synthetic_demo/bundle/bundle.json"))\nINPUT = Path(os.environ.get("SERS_NOTEBOOK_INPUT", PROJECT / "output/synthetic_demo/prepared/Enwave/test.npz"))\nprint("Create these synthetic artifacts first with: sers-pipeline demo --output output/synthetic_demo")'
            ),
            md("### 2. Execute the cascade"),
            code(
                'report = None\nif BUNDLE.is_file() and INPUT.is_file():\n    report = run_pipeline(BUNDLE, INPUT, WORK / "prediction", batch_size=4)\n    print({k: report[k] for k in ("source_device", "rows", "contains_untrained_models")})\nelse:\n    print("Bundle/input unavailable. Run the demo, or select your verified research artifacts, then rerun.")'
            ),
            md("## Checks"),
            code(
                'if report is not None:\n    assert (WORK / "prediction/translated.npz").is_file()\n    with np.load(WORK / "prediction/original_predictions.npz") as before, np.load(WORK / "prediction/translated_predictions.npz") as after:\n        assert np.array_equal(before["sample_ids"], after["sample_ids"])\n        assert before["logits"].shape == after["logits"].shape'
            ),
            md(
                "## Next steps\nSynthetic comparison scores do not establish efficacy. For research, use an independently evaluated bundle and record data/model hashes. Translation metrics require an independent Renishaw reference batch."
            ),
        ],
    )
    write(
        "05_explain_predictions.ipynb",
        "# Explicit input-gradient explanation\n\n## Goal\nUnderstand the saliency interface on synthetic inputs without claiming biological interpretation.",
        [
            md("### 1. Explain a fixed logit target"),
            code(
                'import torch\nfrom sersd2d_superraman.classification.model import SuperRaman\nfrom sersd2d_superraman.explainability import explain\ntorch.manual_seed(37)\nmodel = SuperRaman().eval()\nsynthetic(WORK / "synthetic", per_class=5)\nbatch = load_batch(WORK / "synthetic/Enwave.npz")\ngradient, classes = explain(model, batch.spectra[:1], target_classes=np.array([3]), target="logit")\nassert gradient.shape == (1,1400) and np.isfinite(gradient).all()'
            ),
            md("### 2. Display the declared target and axis"),
            code(
                'import matplotlib.pyplot as plt\nfig, axes = plt.subplots(2, 1, figsize=(9,4.8), sharex=True, layout="constrained")\naxes[0].plot(batch.spectra[0], color="#176b87")\naxes[0].set_ylabel("Synthetic intensity")\naxes[0].set_title("Untrained model: gradient of class-3 logit on a synthetic spectrum")\naxes[1].plot(gradient[0], color="#9b3b54")\naxes[1].set_ylabel("Logit input gradient")\naxes[1].set_xlabel("Spectral bin index (physical Raman axis unavailable)")\nfig.savefig(WORK / "saliency.png", dpi=150)\nplt.show()'
            ),
            md(
                "## Checks and next steps\nGradients are with respect to classifier inputs scaled to [-1,1]. For a verified model, choose logit or probability targets and record class IDs; SmoothGrad uses fixed targets and a recorded noise seed. Supply an actual Raman axis before showing cm^-1. SHAP and paper BCE-based explanations are separately deferred extensions."
            ),
        ],
    )
    print("Built five safe-default package notebooks")


if __name__ == "__main__":
    main()
