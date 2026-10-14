"""Execute tutorials with this Python kernel, retaining HTML and plot evidence."""

from pathlib import Path
import base64
import json
import os
import sys
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / "tmp/second_pass_notebooks"
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        book = nbformat.read(path, as_version=4)
        nbformat.validate(book)
        manager = KernelManager(kernel_name="python3")
        manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
        client = NotebookClient(book, km=manager, timeout=600, resources={"metadata": {"path": str(ROOT)}})
        env = os.environ.copy()
        env["SERS_WORKFLOW_CONFIG"] = str(ROOT / "configs/workflow_demo.json")
        env["SERS_WORKFLOW_OUTPUT"] = str(output / "workflow")
        env["PYTHONPATH"] = str(ROOT / "src")
        env["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"
        try:
            book = client.execute(env=env)
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
        nbformat.write(book, output / path.name)
        # Published notebooks retain verified synthetic outputs for immediate review.
        nbformat.write(book, path)
        html, _ = HTMLExporter().from_notebook_node(book)
        (output / path.with_suffix(".html").name).write_text(html, encoding="utf-8")
        images = []
        for i, cell in enumerate(book.cells):
            for output_index, item in enumerate(cell.get("outputs", [])):
                image = item.get("data", {}).get("image/png")
                if image:
                    dest = output / f"{path.stem}_cell{i}_output{output_index}.png"
                    dest.write_bytes(base64.b64decode(image))
                    images.append(dest.name)
        records.append(
            {"notebook": path.name, "executed": True, "cells": len(book.cells), "plot_images": images}
        )
        print(path.name, "executed")
    (output / "validation.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
