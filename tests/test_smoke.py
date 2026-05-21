"""Stage 0 smoke tests — verify repo structure and imports only."""

import importlib
import pathlib


def test_src_packages_importable():
    packages = [
        "src",
        "src.data",
        "src.models",
        "src.models.kan",
        "src.losses",
        "src.metrics",
        "src.utils",
    ]
    for pkg in packages:
        mod = importlib.import_module(pkg)
        assert mod is not None, f"Failed to import {pkg}"


def test_required_directories_exist():
    root = pathlib.Path(__file__).parent.parent
    required = [
        "src/data",
        "src/models/kan",
        "src/losses",
        "src/metrics",
        "src/utils",
        "configs/data",
        "configs/model",
        "configs/loss",
        "configs/experiment",
        "tests",
        "scripts",
        "reports/figures",
        "reports/tables",
        ".claude/agents",
        ".claude/commands",
        ".claude/skills",
    ]
    for rel in required:
        path = root / rel
        assert path.is_dir(), f"Expected directory not found: {rel}"


def test_required_root_files_exist():
    root = pathlib.Path(__file__).parent.parent
    required = [
        "CLAUDE.md",
        "AGENTS.md",
        "README.md",
        "TODO.md",
        "pyproject.toml",
        ".gitignore",
    ]
    for rel in required:
        path = root / rel
        assert path.is_file(), f"Expected file not found: {rel}"


def test_claude_md_contains_all_rules():
    root = pathlib.Path(__file__).parent.parent
    text = (root / "CLAUDE.md").read_text()
    for i in range(1, 11):
        assert f"R{i}" in text, f"CLAUDE.md missing rule R{i}"


def test_todo_all_unchecked():
    root = pathlib.Path(__file__).parent.parent
    text = (root / "TODO.md").read_text()
    assert "- [x]" not in text.lower(), "TODO.md should have no checked items at Stage 0"
    assert "- [ ]" in text, "TODO.md should contain unchecked items"
