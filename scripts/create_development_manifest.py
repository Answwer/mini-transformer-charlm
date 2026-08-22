"""Create a deterministic work-level development-validation manifest."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


def build_manifest(report: dict[str, Any], development_count: int = 3) -> dict[str, Any]:
    if development_count <= 0:
        raise ValueError("development_count must be positive")
    splits = report.get("splits", {})
    works = splits.get("works", {})
    train_titles = list(works.get("train", []))
    if len(train_titles) <= development_count:
        raise ValueError("the official train split is too small for development holdout")
    records = report.get("work_records", [])
    if not records:
        raise ValueError("format report is missing work_records")
    train_records = [record for record in records if record.get("split") == "train"]
    if len(train_records) != len(train_titles):
        raise ValueError("train work list and train work_records have different lengths")

    # Evenly spaced records make the holdout deterministic while sampling the
    # beginning, middle, and end of the official train works.
    selected_positions = [
        round((index + 1) * (len(train_records) - 1) / (development_count + 1))
        for index in range(development_count)
    ]
    selected_positions = sorted(set(selected_positions))
    if len(selected_positions) != development_count:
        raise ValueError("development holdout selection produced duplicate positions")
    selected_titles = {train_records[position]["title"] for position in selected_positions}

    derived = copy.deepcopy(report)
    derived["development_validation"] = {
        "source_report_sha256": None,
        "development_count": development_count,
        "selected_titles": [train_records[position]["title"] for position in selected_positions],
        "selection_rule": "evenly spaced records from official train works",
    }
    derived["splits"] = copy.deepcopy(splits)
    derived["splits"]["works"] = {
        "train": [title for title in train_titles if title not in selected_titles],
        "development_validation": [
            train_records[position]["title"] for position in selected_positions
        ],
        "validation": list(works.get("validation", [])),
        "test": list(works.get("test", [])),
    }
    for record in derived["work_records"]:
        if record.get("title") in selected_titles:
            record["split"] = "development_validation"
    derived["splits"]["split_policy"] = {
        "train": len(derived["splits"]["works"]["train"]),
        "development_validation": development_count,
        "validation": len(derived["splits"]["works"]["validation"]),
        "test": len(derived["splits"]["works"]["test"]),
        "unit": "work",
        "official_report_unchanged": True,
    }
    return derived


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--development-count", type=int, default=3)
    args = parser.parse_args()

    report_path = Path(args.report)
    output_path = Path(args.output)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    derived = build_manifest(report, args.development_count)
    derived["development_validation"]["source_report_sha256"] = hashlib.sha256(
        report_path.read_bytes()
    ).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(derived, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(derived["development_validation"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
