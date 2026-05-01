import shutil
from pathlib import Path

src = Path("data")
dst = Path("ClavaDDPM") / "data"

if not src.exists():
    raise FileNotFoundError(f"Source directory does not exist: {src}")

# Copy data/ → ClavaDDPM/data
shutil.copytree(src, dst, dirs_exist_ok=True)

print(f"Copied {src} → {dst}")
