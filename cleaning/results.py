from __future__ import annotations

import csv
import math
from pathlib import Path

from .data import Cell, digest, read_json, write_csv, write_json
from .detection import Prediction


DETECTION_HEADER = ["row_id", "column", "dirty_value", "score", "detector_score", "error"]


def write_predictions(directory, table, predictions, rules_sha256=None, *, evidence_sha256=None):
    csv_file = directory / "detections.csv"
    write_csv(csv_file, DETECTION_HEADER, ([p.cell.row_id, p.cell.column, p.dirty_value,
                                           p.score, p.detector_score, int(p.error)] for p in predictions))
    write_json(directory / "detections.manifest.json", {
        "schema": 2 if evidence_sha256 is not None else 1,
        "dirty_sha256": table.sha256, "prediction_sha256": digest(csv_file.read_bytes()),
        "columns": table.columns, "row_count": len(table.rows),
        "rules_sha256": rules_sha256,
        "evidence_sha256": evidence_sha256,
    })


def read_predictions(directory: Path, table, rules_sha256=None, *, evidence_sha256=None, allow_legacy=False):
    manifest = read_json(directory / "detections.manifest.json")
    file = directory / "detections.csv"
    schema = manifest.get("schema")
    if (schema not in (1, 2) or manifest.get("dirty_sha256") != table.sha256
            or manifest.get("columns") != list(table.columns) or manifest.get("row_count") != len(table.rows)
            or manifest.get("prediction_sha256") != digest(file.read_bytes())):
        raise ValueError("Detection output does not match the dirty snapshot or was modified")
    if rules_sha256 is not None and manifest.get("rules_sha256") != rules_sha256:
        raise ValueError("Detection output uses different rule and evidence settings")
    if evidence_sha256 is not None:
        if schema == 1 and not allow_legacy:
            raise ValueError("Detection output has no evidence fingerprint. Rerun detection or explicitly accept a legacy artifact.")
        if schema == 2 and manifest.get("evidence_sha256") != evidence_sha256:
            raise ValueError("Detection output uses different evidence settings or code")
    records, seen = [], set()
    with file.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != DETECTION_HEADER:
            raise ValueError("Invalid detection CSV header")
        for row in reader:
            cell = Cell(int(row["row_id"]), row["column"])
            if cell in seen or row["dirty_value"] != table.at(cell) or row["error"] not in {"0", "1"}:
                raise ValueError(f"Invalid or duplicated detection: {cell}")
            seen.add(cell)
            score, detector_score = float(row["score"]), float(row["detector_score"])
            if not all(math.isfinite(value) and 0 <= value <= 1 for value in (score, detector_score)):
                raise ValueError(f"Invalid detection score: {cell}")
            records.append(Prediction(cell, row["dirty_value"], score, detector_score, row["error"] == "1"))
    return records


def write_repaired(directory, table, decisions):
    edits = {}
    for item in decisions:
        key = item.cell
        raw = table.at(key)
        if key in edits or raw != item.dirty_value or (not item.accepted and item.repaired_value != raw):
            raise ValueError(f"Invalid repair decision: {key}")
        edits[key] = item.repaired_value
    write_csv(directory / "repaired.csv", table.columns,
              ([edits.get(Cell(rid, col), value) for col, value in zip(table.columns, row)]
               for rid, row in enumerate(table.rows)))
    return edits


