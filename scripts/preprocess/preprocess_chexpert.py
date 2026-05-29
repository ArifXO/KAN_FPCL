"""Resize + JPEG re-encode CheXpert into a fixed-side cache (Stage 9).

Reads `Path` entries from a CheXpert manifest (train.csv / valid.csv),
opens each image, converts to single-channel L, resizes to ``--size``,
and writes the result under ``--dst`` mirroring the source directory
layout.

The script is idempotent: if the destination file already exists with the
target side length it is skipped. Run interruption is therefore safe —
re-running picks up where it stopped, never overwrites, and never silently
corrupts a partial file (writes go through a ``.tmp`` then ``replace``).

Usage:
    python -m scripts.preprocess.preprocess_chexpert \\
        --src /data/chexpert \\
        --dst /data/chexpert_224 \\
        --size 224 \\
        --quality 90 \\
        --manifest train.csv

R8 NOTE: This is a preprocessing step, not a training run, so it does not
emit a `checkpoints/run_*` artifact bundle. The source manifest and dst
tree together are the reproducibility unit; record the git hash next to
the destination directory if you cite preprocessed images in a paper.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

from PIL import Image


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Resize + JPEG re-encode CheXpert.")
    p.add_argument("--src", required=True, type=Path, help="CheXpert root.")
    p.add_argument("--dst", required=True, type=Path, help="Output root.")
    p.add_argument("--manifest", default="train.csv", help="Manifest CSV name.")
    p.add_argument("--size", type=int, default=224, help="Target side (px).")
    p.add_argument("--quality", type=int, default=90, help="JPEG quality.")
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit (debug / dry-run).",
    )
    return p


def _process_one(src_path: Path, dst_path: Path, size: int, quality: int) -> str:
    """Return one of {'skipped', 'written', 'missing'}."""
    if not src_path.exists():
        return "missing"
    if dst_path.exists():
        try:
            with Image.open(dst_path) as im:
                if im.size == (size, size):
                    return "skipped"
        except (OSError, Image.DecompressionBombError) as exc:
            # Existing dst is unreadable/partial — re-encode it below rather
            # than skip, but surface why (R9: no silent failure).
            print(
                f"[preprocess] re-encoding unreadable destination "
                f"{dst_path}: {type(exc).__name__}: {exc}",
                flush=True,
            )
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst_path.with_suffix(dst_path.suffix + ".tmp")
    with Image.open(src_path) as im:
        out = im.convert("L")
        if out.size != (size, size):
            out = out.resize((size, size), Image.BILINEAR)
        out.save(tmp_path, format="JPEG", quality=quality, optimize=True)
    tmp_path.replace(dst_path)
    return "written"


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.size <= 0:
        raise ValueError(f"--size must be > 0, got {args.size}.")
    if not (1 <= args.quality <= 100):
        raise ValueError(f"--quality must be in [1, 100], got {args.quality}.")

    manifest = args.src / args.manifest
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    counts = {"written": 0, "skipped": 0, "missing": 0}
    t0 = time.time()

    with open(manifest, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        if "Path" not in reader.fieldnames:
            raise ValueError(
                f"Manifest {manifest} missing required 'Path' column."
            )
        for i, row in enumerate(reader):
            if args.limit is not None and i >= args.limit:
                break
            rel = Path(row["Path"])
            src_path = args.src / rel
            dst_path = args.dst / rel
            status = _process_one(src_path, dst_path, args.size, args.quality)
            counts[status] += 1
            if (i + 1) % 1000 == 0:
                print(
                    f"[{i + 1}] written={counts['written']} "
                    f"skipped={counts['skipped']} missing={counts['missing']} "
                    f"elapsed={time.time() - t0:.1f}s",
                    flush=True,
                )

    print(
        f"Done. written={counts['written']} skipped={counts['skipped']} "
        f"missing={counts['missing']} total_time={time.time() - t0:.1f}s"
    )
    if counts["missing"] > 0:
        print(
            f"WARNING: {counts['missing']} manifest paths had no source file. "
            "Investigate before training (R9 — no silent gaps)."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
