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
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def public_files():
    paths = [ROOT / name for name in TOP] + [ROOT / "docs" / name for name in DOCS]
    for folder, suffixes in {
        "src": (".py",),
        "tests": (".py", ".json", ".npz"),
        "configs": (".json",),
        "notebooks": (".ipynb",),
        "tools": (".py", ".txt"),
        "examples": (".py", ".md"),
        "experiments": (".md",),
        ".github": (".yml",),
        "legacy": (".md",),
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

    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    prefix = f"SERSD2D-SuperRaman-{version}"
    archive = dist / f"{prefix}-github-source.zip"
    paths = public_files()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            z.write(p, f"{prefix}/{p.relative_to(ROOT).as_posix()}")
    with zipfile.ZipFile(archive) as z:
        for p in paths:
            name = f"{prefix}/{p.relative_to(ROOT).as_posix()}"
            if z.read(name) != p.read_bytes():
                raise ValueError(f"Source ZIP mismatch: {name}")
    wheel = next(dist.glob(f"sersd2d_superraman-{version}-*.whl"))
    with zipfile.ZipFile(wheel) as z:
        for p in (ROOT / "src/sersd2d_superraman").rglob("*.py"):
            if (
                "__pycache__" not in p.parts
                and z.read(p.relative_to(ROOT / "src").as_posix()) != p.read_bytes()
            ):
                raise ValueError(f"Wheel/source mismatch: {p.name}")
        if any("prismsers" in name or "SuperRamanNet/" in name for name in z.namelist()):
            raise ValueError("Wheel contains a legacy/neighbor dependency")
    sdist = next(dist.glob(f"sersd2d_superraman-{version}.tar.gz"))
    with tarfile.open(sdist) as t:
        names = t.getnames()
        if any(
            "/SERSD2DNet/" in name
            or "/SuperRamanNet/" in name
            or "/docs/audit/" in name
            or name.endswith(".pt")
            or name.endswith(".h5")
            for name in names
        ):
            raise ValueError("Source distribution includes private research artifacts")
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
    (dist / "release_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
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
