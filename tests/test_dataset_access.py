"""Offline discovery and first-use input errors for the restricted dataset."""

import json
from pathlib import Path
import pytest
from sersd2d_superraman.cli import main, prepare
from sersd2d_superraman.data.access import DATASET_URL, dataset_info, require_dataset, require_prepared
from sersd2d_superraman.provenance import read_json
from sersd2d_superraman.workflow import Workflow


def test_dataset_info_cli_needs_no_output_or_network(capsys):
    main(["dataset-info"])
    info = json.loads(capsys.readouterr().out)
    assert info["record_url"] == "https://zenodo.org/records/23372983"
    assert info["doi"] == "10.5281/zenodo.23372983"
    assert info["version"] == "1.0.0" and info["access"] == "restricted"
    assert info["files_included_in_git"] is False


def test_missing_dataset_points_to_request_and_creates_no_output(tmp_path):
    output = tmp_path / "prepared"
    with pytest.raises(FileNotFoundError) as error:
        prepare(tmp_path / "not-downloaded.h5", output)
    assert DATASET_URL in str(error.value)
    assert "dataset-info" in str(error.value)
    assert not output.exists()


def test_cli_missing_dataset_explains_local_setup(tmp_path):
    with pytest.raises(SystemExit) as error:
        main(["prepare", "--dataset", str(tmp_path / "missing.h5"), "--output", str(tmp_path / "output")])
    assert DATASET_URL in str(error.value)
    assert "local preprocessed HDF5" in str(error.value)
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("filename", ["approved.zip", "approved.tar.gz", "approved.7z"])
def test_downloaded_archives_need_extraction_before_preparation(tmp_path, filename):
    archive = tmp_path / filename
    archive.write_bytes(b"preflight stand-in; not a real archive")
    with pytest.raises(ValueError, match="Extract the approved dataset archive"):
        require_dataset(archive)


def test_record_url_is_not_treated_as_local_dataset():
    with pytest.raises(ValueError, match="local downloaded path"):
        require_dataset(DATASET_URL)


def test_canonical_directory_lists_missing_devices_before_creating_output(tmp_path):
    directory = tmp_path / "canonical"
    directory.mkdir()
    (directory / "Renishaw.npz").write_bytes(b"preflight stand-in")
    with pytest.raises(FileNotFoundError, match="Enwave.npz"):
        prepare(directory, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_single_npz_is_not_a_five_device_preparation_input(tmp_path):
    path = tmp_path / "Enwave.npz"
    path.write_bytes(b"preflight stand-in")
    with pytest.raises(ValueError, match="all five devices"):
        require_dataset(path)


def test_downloaded_directory_is_not_mistaken_for_prepared_study(tmp_path):
    with pytest.raises(FileNotFoundError, match="output of sers-pipeline prepare"):
        require_prepared(tmp_path)


@pytest.mark.parametrize("mode", ["dataset", "prepared_data"])
def test_research_workflow_missing_input_has_actionable_access_guidance(tmp_path, mode):
    config = read_json(Path(__file__).parents[1] / "configs/workflow_research.json")
    config.pop("prepared_data")
    config[mode] = "missing-local-input"
    config["run_directory"] = "run"
    workflow = Workflow(tmp_path, config)
    with pytest.raises(FileNotFoundError) as error:
        workflow.prepare()
    assert DATASET_URL in str(error.value)
    assert workflow.status() == {}
    assert not (workflow.output / "prepared").exists()


def test_existing_local_hdf5_path_passes_discovery_without_claiming_content_validation(tmp_path):
    path = tmp_path / "downloaded.h5"
    path.write_bytes(b"path-only preflight; content validation belongs to the loader")
    assert require_dataset(path) == path
    assert dataset_info()["local_input_formats"]
