"""Compare CSV, JSONL or grouped legacy-rule JSON by explicit record keys."""

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


def records(path):
    if path.suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, strict=True)
            if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
                raise ValueError(f"Missing or duplicate headers in {path.name}")
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f"Malformed CSV row in {path.name} at line {reader.line_num}")
                yield row
    elif path.suffix == ".jsonl":
        with path.open(encoding="utf-8-sig") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
    elif path.suffix == ".json":
        document = json.loads(path.read_text(encoding="utf-8-sig"))
        yield from (rule for group in document["rules"].values() for rule in group)
    else:
        raise ValueError(f"Unsupported input format: {path.suffix}")


def align(path, keys, fields, row_map=None, row_key=None, pick=None, include_all=False):
    """Keep original row order and reject duplicate keys for one-to-one joins."""
    found = {}
    for row in records(path):
        if row_map is not None:
            rid = int(row[row_key])
            if rid not in row_map:
                continue
            row[row_key] = row_map[rid]
        if pick and not pick(row):
            continue
        key = tuple(str(row[name]) for name in keys)
        if key in found:
            raise ValueError(f"Duplicate key in {path.name}: {key}")
        found[key] = row if include_all else {name: row[name] for name in fields or ()}
    return found


def equal_values(left, right, atol, rtol):
    if atol is None:
        return left == right
    if left == "" and right == "":
        return True
    try:
        a, b = float(left), float(right)
    except (ValueError, TypeError):
        return False
    return math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, abs_tol=atol, rel_tol=rtol)


def compare(left, right, fields=(), *, atol=1e-7, rtol=1e-6, detail=None, ordered=False):
    """Fields are (old name, new name, numeric). Exact comparison is the default."""
    overlap = left.keys() & right.keys()
    union_count = len(left) + len(right) - len(overlap)
    old_only, new_only, mismatched = len(left) - len(overlap), len(right) - len(overlap), 0
    field_counts = Counter()
    numerical = {}
    sequence_equal = list(left) == list(right)
    for key, old in left.items():
        if key not in right:
            if detail:
                detail({"kind": "old_only", "key": key})
            continue
        new = right[key]
        different = False
        for a, b, numeric in fields:
            va, vb = old[a], new[b]
            if numeric:
                stats = numerical.setdefault(a, {"count": 0, "max_abs": 0.0, "sum_abs": 0.0,
                                                  "over_tolerance": 0, "invalid_pairs": 0})
                try:
                    x, y = float(va), float(vb)
                    if not math.isfinite(x) or not math.isfinite(y):
                        raise ValueError("Non-finite numeric pair")
                    error = abs(x-y)
                    stats["count"] += 1
                    stats["sum_abs"] += error
                    stats["max_abs"] = max(stats["max_abs"], error)
                    stats["over_tolerance"] += not math.isclose(x, y, abs_tol=atol, rel_tol=rtol)
                except (ValueError, TypeError):
                    stats["invalid_pairs"] += 1
            if not equal_values(va, vb, atol if numeric else None, rtol):
                field_counts[a] += 1
                different = True
                if detail:
                    detail({"kind": "value", "key": key, "old_field": a, "new_field": b, "old": va, "new": vb})
        mismatched += different
    if detail:
        for key in right:
            if key not in left:
                detail({"kind": "new_only", "key": key})
    for stats in numerical.values():
        absolute_sum = stats.pop("sum_abs")
        stats["mean_abs"] = absolute_sum / stats["count"] if stats["count"] else None
    return {
        "status": "FAIL" if old_only or new_only or mismatched or (ordered and not sequence_equal) else "PASS",
        "old": len(left), "new": len(right), "intersection": len(overlap),
        "old_only": old_only, "new_only": new_only, "value_mismatches": mismatched,
        "key_jaccard": len(overlap) / union_count if union_count else 1.0,
        "agreement": (len(overlap) - mismatched) / union_count if union_count else 1.0,
        "common_value_agreement": (len(overlap) - mismatched) / len(overlap) if overlap else None,
        "same_key_order": sequence_equal, "field_mismatches": dict(field_counts),
        "numeric_errors": numerical, "numeric_tolerances": {"atol": atol, "rtol": rtol},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old", type=Path)
    parser.add_argument("new", type=Path)
    parser.add_argument("--key", action="append", required=True, help="OLD[:NEW], repeat for composite key")
    parser.add_argument("--exact", action="append", default=[], help="OLD[:NEW] column")
    parser.add_argument("--numeric", action="append", default=[], help="OLD[:NEW] numeric feature")
    parser.add_argument("--atol", type=float, default=1e-7)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--ordered", action="store_true")
    parser.add_argument("--details", type=Path, required=True, help="New JSONL path")
    args = parser.parse_args()
    if not all(math.isfinite(x) and x >= 0 for x in (args.atol, args.rtol)):
        parser.error("Tolerances must be finite and nonnegative")

    def sides(spec):
        left, _, right = spec.partition(":")
        return left, right or left

    left_keys, right_keys = zip(*(sides(key) for key in args.key))
    fields = [(*sides(name), numeric) for group, numeric in [(args.exact, False), (args.numeric, True)] for name in group]
    try:
        old = align(args.old, left_keys, [a for a, _, _ in fields])
        new = align(args.new, right_keys, [b for _, b, _ in fields])
        with args.details.open("x", encoding="utf-8") as handle:
            summary = compare(old, new, fields, atol=args.atol, rtol=args.rtol, ordered=args.ordered,
                              detail=lambda row: handle.write(json.dumps(row, ensure_ascii=False) + "\n"))
    except (OSError, KeyError, ValueError, csv.Error) as error:
        summary = {"status": "BLOCKED", "reason": str(error)}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return {"PASS": 0, "FAIL": 1, "BLOCKED": 2}[summary["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
