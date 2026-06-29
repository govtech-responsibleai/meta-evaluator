"""Packaging assertion: the built wheel ships the prebuilt annotator dist."""

import glob
import subprocess
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_wheel_includes_annotator_dist(tmp_path):
    """`uv build --wheel` must bundle annotator/frontend/dist/index.html."""
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(tmp_path)],
        cwd=REPO_ROOT,
        check=True,
    )
    wheels = glob.glob(str(tmp_path / "*.whl"))
    assert wheels, "no wheel produced"
    with zipfile.ZipFile(wheels[0]) as zf:
        names = zf.namelist()
    assert any(
        n.endswith("meta_evaluator/annotator/frontend/dist/index.html") for n in names
    ), f"dist/index.html not in wheel; sample: {names[:20]}"
