"""Build a public source ZIP from a whitelist and verify wheel/source content."""

from pathlib import Path
import hashlib
import json
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TOP = (
    "README.md",
    "LICENSE",
    "NOTICE.md",
    "CITATION.cff",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "pyproject.toml",
    "MANIFEST.in",
    ".gitignore",
    ".gitattributes",
)
DOCS = (
    "PAPER_IMPLEMENTATION_MAP.md",
    "DATA.md",
    "REPRODUCIBILITY.md",
    "MODEL_CARDS.md",
    "BENCHMARKS.md",
    "IMPLEMENTATION_STATUS.md",
    "VERIFICATION.json",
    "NOTEBOOK_GUIDE.md",
    "SECOND_PASS_REVIEW.md",
    "THIRD_PASS_REVIEW.md",
    "THIRD_PASS_VERIFICATION.json",
    "REFERENCES.md",
    "paper_contract.json",
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def public_files():
    paths = [ROOT / name for name in TOP] + [ROOT / "docs" / name for name in DOCS]
    for folder, suffixes in {
        "src": (".py",),
        "tests": (".py", ".json", ".npz"),
        "configs": (".json", ".txt"),
        "notebooks": (".ipynb",),
        "tools": (".py", ".txt"),
        "examples": (".py", ".md"),
        ".github": (".yml",),
    }.items():
        for p in (ROOT / folder).rglob("*"):
            if p.is_file() and p.suffix in suffixes and "local" not in p.relative_to(ROOT).parts:
                paths.append(p)
    for p in paths:
        if not p.is_file():
            raise ValueError(f"Required public release file missing: {p.relative_to(ROOT)}")
    return sorted(set(paths))


def main():
    import tomllib

    version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    prefix = f"SERS-D2DNet-SuperRaman-{version}"
    archive = dist / f"{prefix}-github-source.zip"
    paths = public_files()
    candidates = sorted(dist.glob(f"sersd2d_superraman-{version}-*.whl"))
    if len(candidates) != 1:
        raise ValueError(f"Expected one current wheel; found {len(candidates)}. Build into a clean dist/.")
    wheel = candidates[0]
    with zipfile.ZipFile(wheel) as z:
        for p in (ROOT / "src/sersd2d_superraman").rglob("*.py"):
            if (
                "__pycache__" not in p.parts
                and z.read(p.relative_to(ROOT / "src").as_posix()) != p.read_bytes()
            ):
                raise ValueError(f"Wheel/source mismatch: {p.name}")
        if any("prismsers" in name or "SuperRamanNet/" in name for name in z.namelist()):
            raise ValueError("Wheel contains a legacy/neighbor dependency")
    sdist = dist / f"sersd2d_superraman-{version}.tar.gz"
    if not sdist.is_file():
        raise ValueError("Current source distribution missing; run python -m build first")
    with tarfile.open(sdist) as t:
        names = t.getnames()
        for p in paths:
            matches = [
                m for m in t.getmembers() if m.name.partition("/")[2] == p.relative_to(ROOT).as_posix()
            ]
            if len(matches) != 1 or not matches[0].isfile():
                raise ValueError(f"Source distribution missing public file: {p.relative_to(ROOT)}")
            if t.extractfile(matches[0]).read() != p.read_bytes():
                raise ValueError(f"Source distribution content mismatch: {p.relative_to(ROOT)}")
        if any(
            "/SERSD2DNet/" in name
            or "/SuperRamanNet/" in name
            or "/docs/audit/" in name
            or name.endswith(".pt")
            or name.endswith(".h5")
            for name in names
        ):
            raise ValueError("Source distribution includes private research artifacts")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            z.write(p, f"{prefix}/{p.relative_to(ROOT).as_posix()}")
    with zipfile.ZipFile(archive) as z:
        for p in paths:
            name = f"{prefix}/{p.relative_to(ROOT).as_posix()}"
            if z.read(name) != p.read_bytes():
                raise ValueError(f"Source ZIP mismatch: {name}")
    manifest = {
        "version": version,
        "public_source_files": len(paths),
        "source_and_wheel_match": True,
        "files": [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha(p.read_bytes())} for p in paths],
        "artifacts": [
            {"name": p.name, "bytes": p.stat().st_size, "sha256": sha(p.read_bytes())}
            for p in (archive, wheel, sdist)
        ],
    }
    (dist / "release_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in ("version", "public_source_files", "source_and_wheel_match", "artifacts")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
