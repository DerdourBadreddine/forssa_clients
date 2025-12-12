from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd


def _latest_submission(outputs_dir: Path) -> Path:
    paths = sorted(glob.glob(str(outputs_dir / "submission_*.csv")))
    if not paths:
        raise FileNotFoundError(f"No outputs/submission_*.csv found under {outputs_dir}")
    return Path(paths[-1])


def _validate(test_df: pd.DataFrame, sub_df: pd.DataFrame) -> dict:
    if "id" not in test_df.columns:
        raise AssertionError("test_file.csv must contain column 'id'")
    if "id" not in sub_df.columns or "Class" not in sub_df.columns:
        raise AssertionError("submission must contain columns: id, Class")

    report = {
        "rows_test": int(len(test_df)),
        "rows_sub": int(len(sub_df)),
        "ids_equal_order": bool(test_df["id"].equals(sub_df["id"])),
        "same_id_set": bool(set(test_df["id"]) == set(sub_df["id"])),
        "duplicates": int(sub_df["id"].duplicated().sum()),
        "class_min": int(sub_df["Class"].min()) if len(sub_df) else None,
        "class_max": int(sub_df["Class"].max()) if len(sub_df) else None,
        "class_has_nan": bool(sub_df["Class"].isna().any()),
        "id_has_nan": bool(sub_df["id"].isna().any()),
    }

    # Hard assertions (fail fast)
    if report["rows_test"] != report["rows_sub"]:
        raise AssertionError(f"Row mismatch: test={report['rows_test']} sub={report['rows_sub']}")
    if report["id_has_nan"]:
        raise AssertionError("NaN found in submission id")
    if report["class_has_nan"]:
        raise AssertionError("NaN found in submission Class")
    if report["duplicates"] != 0:
        raise AssertionError(f"Duplicate ids in submission: {report['duplicates']}")
    if not report["same_id_set"]:
        raise AssertionError("Submission id set differs from test id set")
    if not report["ids_equal_order"]:
        raise AssertionError("Submission id order does not match test_file.csv")
    if not sub_df["Class"].between(1, 9).all():
        raise AssertionError("Class must be integers in [1..9]")

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a Kaggle submission against test_file.csv")
    parser.add_argument("--submission", type=Path, default=None, help="Path to submission CSV (default: latest outputs/submission_*.csv)")
    parser.add_argument("--data-dir", type=Path, default=Path("data") / "forsa-clients-satisfaction")
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    test_path = args.data_dir / "test_file.csv"
    if not test_path.exists():
        raise FileNotFoundError(f"Missing test file: {test_path}")

    sub_path = args.submission or _latest_submission(args.outputs_dir)
    if not sub_path.exists():
        raise FileNotFoundError(f"Missing submission file: {sub_path}")

    test_df = pd.read_csv(test_path)
    sub_df = pd.read_csv(sub_path)

    report = _validate(test_df, sub_df)
    print(f"submission_path: {sub_path}")
    for k, v in report.items():
        print(f"{k}: {v}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise SystemExit(1)
