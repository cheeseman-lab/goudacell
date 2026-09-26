"""Vendor brieflow's phenotype modules into ``src/goudacell/brieflow/``.

Each module is copied verbatim from a brieflow checkout at a given commit, with brieflow's
``LICENSE``; the only change is rewriting ``lib.`` imports to ``goudacell.brieflow.``. The
commit is recorded in ``src/goudacell/brieflow/__init__.py``. Never edit the vendored files:
re-run this script.

    python scripts/sync_brieflow.py --brieflow /path/to/brieflow [--ref <commit>] [--check]
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

DEST = Path(__file__).resolve().parents[1] / "src" / "goudacell" / "brieflow"

# Modules under brieflow's workflow/lib, the phenotype notebook's calls and their imports
MODULES = (
    "shared/segment_cellpose.py",
    "shared/segmentation_utils.py",
    "phenotype/identify_cytoplasm_cellpose.py",
    "phenotype/extract_phenotype_cp_emulator.py",
    "phenotype/extract_phenotype_cp_measure.py",
    "phenotype/constants.py",
    "phenotype/custom_features.py",
    "phenotype/segment_secondary_object.py",
    "phenotype/extract_phenotype_second_objs.py",
    "external/cp_emulator.py",
    "shared/feature_extraction.py",
    "shared/feature_table_utils.py",
    "shared/feature_utils.py",
    "shared/log_filter.py",
    "shared/image_utils.py",
)

IMPORT_RE = re.compile(r"^(\s*)(from|import) lib\.", re.MULTILINE)
MODULE_IMPORT_RE = re.compile(r"^(?:from|import) goudacell\.brieflow\.([\w.]+)", re.MULTILINE)

INIT_TEMPLATE = (
    '"""Brieflow\'s phenotype code, vendored by scripts/sync_brieflow.py; do not edit."""\n'
    "\n"
    'BRIEFLOW_COMMIT = "{commit}"\n'
    "BRIEFLOW_MODULES = (\n{modules})\n"
)


def rewrite_imports(text: str) -> str:
    """Rewrite ``from lib.``/``import lib.`` statements to the vendored package."""
    return IMPORT_RE.sub(r"\1\2 goudacell.brieflow.", text)


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout


def resolve_commit(root: Path, ref: str = "HEAD") -> str:
    """Full commit SHA of ``ref`` in the brieflow checkout at ``root``."""
    return _git(root, "rev-parse", f"{ref}^{{commit}}").strip()


def vendored_sources(root: Path, ref: str = "HEAD") -> dict:
    """The vendored file contents, keyed by path relative to ``src/goudacell/brieflow``.

    Args:
        root: A brieflow git checkout (any working-tree state; files are read at ``ref``).
        ref: Commit to vendor.

    Returns:
        ``{relative_path: text}`` for every module, brieflow's ``LICENSE`` and the generated
        ``__init__.py`` files.
    """
    commit = resolve_commit(root, ref)
    files = {
        module: rewrite_imports(_git(root, "show", f"{commit}:workflow/lib/{module}"))
        for module in MODULES
    }

    # Every module-level import must resolve to a vendored module
    vendored = {m.removesuffix(".py").replace("/", ".") for m in MODULES}
    for module, text in files.items():
        missing = set(MODULE_IMPORT_RE.findall(text)) - vendored
        if missing:
            raise RuntimeError(f"{module} imports unvendored brieflow modules: {sorted(missing)}")

    files["LICENSE"] = _git(root, "show", f"{commit}:LICENSE")

    listed = "".join(f'    "{m}",\n' for m in MODULES)
    files["__init__.py"] = INIT_TEMPLATE.format(commit=commit, modules=listed)
    for package in sorted({str(Path(m).parent) for m in MODULES}):
        files[f"{package}/__init__.py"] = ""
    return files


def main(argv=None) -> int:
    """Sync (or with ``--check``, verify) the vendored brieflow modules."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--brieflow", required=True, type=Path, help="brieflow git checkout")
    parser.add_argument("--ref", default="HEAD", help="commit to vendor (default: HEAD)")
    parser.add_argument("--check", action="store_true", help="only report differences")
    args = parser.parse_args(argv)

    files = vendored_sources(args.brieflow.expanduser().resolve(), args.ref)
    stale = [
        path for path, text in files.items()
        if not (DEST / path).is_file() or (DEST / path).read_text() != text
    ]
    present = [str(p.relative_to(DEST)) for p in DEST.rglob("*.py")] if DEST.is_dir() else []
    extra = sorted(path for path in present if path not in files)

    if args.check:
        for path in stale:
            print(f"differs: {path}")
        for path in extra:
            print(f"not vendored: {path}")
        return 1 if stale or extra else 0

    for path in extra:
        (DEST / path).unlink()
    for path in stale:
        (DEST / path).parent.mkdir(parents=True, exist_ok=True)
        (DEST / path).write_text(files[path])
    print(f"Vendored {len(MODULES)} modules at {resolve_commit(args.brieflow, args.ref)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
