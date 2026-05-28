"""Copy auto-generated tables from runs/tables/ to reports/tables/ for git commit."""

import shutil
from pathlib import Path

src = Path("runs/tables")
dst = Path("reports/tables")
dst.mkdir(parents=True, exist_ok=True)
for f in src.glob("table_h*.md"):
    shutil.copy2(f, dst / f.name)
    print(f"Published {f.name} -> reports/tables/")
