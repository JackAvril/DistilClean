from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import tempfile
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, encoding="utf-8", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_csv(path: Path, header, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


@dataclass(frozen=True, order=True)
class Cell:
    row_id: int
    column: str


@dataclass
class Table:
    columns: tuple[str, ...]
    rows: list[tuple[str, ...]]
    sha256: str

    def __post_init__(self):
        self.index = {name: i for i, name in enumerate(self.columns)}

    def at(self, cell: Cell) -> str:
        if not 0 <= cell.row_id < len(self.rows) or cell.column not in self.index:
            raise ValueError(f"Cell outside dirty table: {cell}")
        return self.rows[cell.row_id][self.index[cell.column]]

    @classmethod
    def load(cls, path: Path):
        payload = path.read_bytes()
        stream = io.StringIO(payload.decode("utf-8-sig"), newline="")
        reader = csv.reader(stream, strict=True)
        columns = tuple(next(reader, ()))
        if not columns or any(not col.strip() for col in columns) or len(set(columns)) != len(columns):
            raise ValueError(f"Missing or duplicate CSV headers in {path}")
        rows = []
        for index, row in enumerate(reader):
            if len(row) != len(columns):
                raise ValueError(f"CSV data row {index} has {len(row)} cells, expected {len(columns)}")
            rows.append(tuple(row))
        return cls(columns, rows, digest(payload))


NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)")
TIME = re.compile(
    r"(?<![\d:])(\d{1,2}):(\d{2})(?!\d)\s*([ap])\.?\s*m\.?",
    re.IGNORECASE,
)
FULL_TIME = re.compile(r"(\d{1,2}):(\d{2})")
MARKED_TIME = re.compile(
    r"(?<![\d:])(\d{1,2})\s*:\s*(\d{2})(?!\d)\s*([ap])(?:\.?m\.?|(?=\b|[a-z]))",
    re.I,
)
UNMARKED_TIME = re.compile(r"(?<![\d:])(\d{1,2})\s*:\s*(\d{2})(?![\d:])")


def number(value: str):
    if not NUMBER.fullmatch(value.strip()):
        return None
    val = float(value)
    return val if math.isfinite(val) else None


def time_minutes(value: str, *, extract=False):
    text = value.strip()
    match = TIME.fullmatch(text)
    if match:
        hour, minute, part = match.groups()
        if 1 <= int(hour) <= 12 and int(minute) < 60:
            return (int(hour) % 12 + 12 * (part.lower() == "p")) * 60 + int(minute)
        return None
    match = FULL_TIME.fullmatch(text)
    if match and int(match[1]) < 24 and int(match[2]) < 60:
        return int(match[1]) * 60 + int(match[2])
    if extract:
        marked_matches = MARKED_TIME.findall(text)
        marked = {(int(hour) % 12 + 12 * (part.lower() == "p")) * 60 + int(minute)
                  for hour, minute, part in marked_matches
                  if 1 <= int(hour) <= 12 and int(minute) < 60}
        if marked_matches:
            return next(iter(marked)) if len(marked) == 1 else None
        bare = {int(hour) * 60 + int(minute) for hour, minute in UNMARKED_TIME.findall(text)
                if int(hour) < 24 and int(minute) < 60}
        if bare:
            return next(iter(bare)) if len(bare) == 1 else None
        if re.fullmatch(r"\d{3,4}", text):
            hour, minute = int(text[:-2]), int(text[-2:])
            if hour < 24 and minute < 60:
                return hour * 60 + minute
    return None


def shape(value: str) -> str:
    runs = ("A" if char.isalpha() else "D" if char.isdigit() else char for char in value)
    return "".join(runs)


KINDS = {"text", "category", "integer", "number", "percent", "time", "issn"}


@dataclass
class Field:
    kind: str
    required: bool
    domain: dict[str, str]
    pattern: object
    bounds: tuple
    aliases: dict
    missing: frozenset
    focus: bool

    def key(self, value: str) -> str:
        key = " ".join(value.split()).casefold()
        return "" if key in self.missing else key

    def numeric(self, value: str):
        if self.kind == "time":
            return time_minutes(value)
        if self.kind in {"integer", "number", "percent"}:
            text = value.strip()
            return number(text.removesuffix("%") if self.kind == "percent" else text)
        return None

    def issues(self, value: str) -> tuple[str, ...]:
        key = self.key(value)
        if not key:
            return ("required",) if self.required else ()
        problems = []
        if self.domain and key not in self.domain:
            problems.append("domain")
        if self.pattern and not self.pattern.fullmatch(value.strip()):
            problems.append("pattern")
        if self.kind in {"integer", "number", "percent", "time"}:
            val = self.numeric(value)
            low, high = self.bounds
            if val is None or (self.kind == "integer" and not val.is_integer()):
                problems.append("type")
            elif (low is not None and val < low) or (high is not None and val > high):
                problems.append("range")
        if self.kind == "issn":
            match = re.fullmatch(r"(\d{4})-(\d{3})([0-9Xx])(?:\s*\(print\))?", value.strip(), re.I)
            if not match:
                problems.append("type")
            else:
                chars = "".join(match.groups()).upper()
                digits = [10 if char == "X" else int(char) for char in chars]
                if sum(weight * digit for weight, digit in zip(range(8, 0, -1), digits)) % 11:
                    problems.append("checksum")
        return tuple(problems)

    def transforms(self, value: str):
        """Generate deterministic alternatives using field types and declared aliases."""
        key = self.key(value)
        if key in self.aliases:
            yield self.aliases[key]
        if not key:
            return
        if self.kind == "time":
            minute = time_minutes(value, extract=True)
            if minute is not None:
                hour, minute = divmod(minute, 60)
                yield f"{hour % 12 or 12}:{minute:02d} {'a.m.' if hour < 12 else 'p.m.'}"
        elif self.kind == "issn":
            match = re.fullmatch(r"(\d{4})[- ]?(\d{3}[\dXx])(?:\s*\(print\))?", value.strip(), re.I)
            if match:
                yield f"{match[1]}-{match[2].upper()}"
        elif self.kind in {"integer", "number", "percent"}:
            numeric_text = value.strip().removesuffix("%").strip()
            val = number(numeric_text)
            if val is not None and (self.kind != "integer" or val.is_integer()):
                compact = format(Decimal(numeric_text), "f")
                if "." in compact:
                    compact = compact.rstrip("0").rstrip(".")
                yield compact
                if self.kind == "percent" and value.strip().endswith("%"):
                    yield compact + "%"
                    yield format(Decimal(numeric_text) / 100, "f")
        if value.strip() != value:
            yield value.strip()


def infer_kind(values, missing):
    present = [value.strip() for value in values if value.strip().casefold() not in missing]
    if not present:
        return "text"
    counts = [number(value) for value in present]
    if all(val is not None for val in counts):
        # Digit identifiers with leading zeroes remain textual.
        if any(re.fullmatch(r"0\d+", value) for value in present):
            return "text"
        return "integer" if all(val.is_integer() for val in counts) else "number"
    if all(time_minutes(value) is not None for value in present):
        return "time"
    return "category" if len(set(present)) <= 20 else "text"


def fields_for(table: Table, config: dict):
    specs = config.get("fields", {})
    unknown = set(specs) - set(table.columns)
    if unknown:
        raise ValueError(f"Configured columns absent in dirty table: {sorted(unknown)}")
    base_missing = frozenset(str(value).casefold() for value in config.get("missing", [""]))
    result = {}
    for col in table.columns:
        spec = specs.get(col, {})
        missing = (base_missing | frozenset(x.casefold() for x in spec.get("missing", []))) | {""}
        kind = spec.get("kind")
        if kind is None:
            index = table.index[col]
            kind = infer_kind((row[index] for row in table.rows[:1024]), missing)
        if kind not in KINDS:
            raise ValueError(f"Unknown field kind: {kind}")
        result[col] = Field(
            kind, spec.get("required", False),
            {" ".join(value.split()).casefold(): value for value in spec.get("domain", [])},
            re.compile(spec["pattern"], re.I) if "pattern" in spec else None,
            tuple(spec.get("bounds", (None, None))),
            {key.casefold(): val for key, val in spec.get("aliases", {}).items()},
            missing, spec.get("target", True),
        )
    return result
