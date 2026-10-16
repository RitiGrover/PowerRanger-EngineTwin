"""Download LiU-ICE data and build a runtime-compatible measured guard bundle."""

from __future__ import annotations

import sys
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from enginetwin.liu_guard import SCENARIOS  # noqa: E402
from enginetwin.liu_ice import (  # noqa: E402
    SOURCE_URL,
    _group_split,
    build_feature_windows,
    fit_sensor_consistency_models,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "liu_ice"
RAW = DATA / "raw" / "trainingdata"
ARCHIVE = DATA / "trainingdata.zip"
FEATURES = DATA / "processed_features.csv.gz"
BUNDLE = DATA / "guard_bundle.joblib"


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            name = PurePosixPath(member.filename)
            target = (destination / Path(*name.parts)).resolve()
            if name.is_absolute() or ".." in name.parts or (target != root and root not in target.parents):
                raise ValueError(f"Unsafe archive path: {member.filename}")
        zipped.extractall(destination)


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    if not FEATURES.exists():
        if not list(RAW.glob("*.csv")):
            if not ARCHIVE.exists():
                urllib.request.urlretrieve(SOURCE_URL, ARCHIVE)
            safe_extract(ARCHIVE, DATA / "raw")
        build_feature_windows(RAW).to_csv(FEATURES, index=False, compression="gzip")
    windows = pd.read_csv(FEATURES)
    train_mask, test_mask = _group_split(windows)
    consistency = fit_sensor_consistency_models(windows.loc[train_mask])
    held_out = windows.loc[test_mask & windows["trajectory_id"].isin(SCENARIOS)].copy()
    if set(held_out["trajectory_id"].unique()) != set(SCENARIOS):
        raise RuntimeError("The deterministic held-out split does not contain every dashboard scenario.")
    joblib.dump(
        {
            "consistency": consistency,
            "test_windows": held_out,
            "source": "LiU-ICE held-out replay",
            "model_runtime": "rebuilt with the installed scikit-learn version",
        },
        BUNDLE,
        compress=3,
    )
    print(f"Saved {len(held_out):,} held-out windows to {BUNDLE}")


if __name__ == "__main__":
    main()
