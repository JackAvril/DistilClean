"""Deterministic repairs declared as field operations on the dirty snapshot."""

import math
import re

from .data import Cell


def replacement(operation, raw):
    value = raw.strip()
    kind = operation["kind"]
    if "requires" in operation and operation["requires"] not in value:
        return None
    if kind in {"integer_suffix", "digit_suffix"}:
        text = value.lower()
        if operation.get("remove_spaces"):
            text = text.replace(" ", "")
        if text in operation.get("empty_tokens", ["", "empty"]):
            return None
        if operation.get("canonical_pattern") and re.fullmatch(operation["canonical_pattern"], text):
            return text
        if kind == "digit_suffix":
            if not any(token in text for token in operation["keywords"]):
                return None
            text = "".join(char for char in text if char.isdigit())
        else:
            for before, after in operation.get("replacements", {}).items():
                text = text.replace(before, after)
            text = text.strip()
        if not text.isdigit():
            return None
        numeric = int(text)
        low, high = operation.get("bounds", (None, None))
        if (low is not None and numeric < low) or (high is not None and numeric > high):
            return None
        return str(numeric) + operation["suffix"]
    if kind == "trailing_code":
        match = re.search(r"\b([a-z]{2})\b$", value, re.I)
        return match[1].upper() if match and match[1].casefold() in operation["codes"] else None
    if kind == "missing_default":
        return operation["value"] if value.casefold() in operation["tokens"] else None
    if kind == "integer_float":
        match = re.fullmatch(r"(\d+)\.0", value)
        if match and int(match[1]) > operation["exclusive_min"]:
            return str(int(match[1]))
        return None
    if kind == "rotate_mdy":
        match = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{2}|\d{4})", value)
        if match:
            month, day = map(int, match.groups()[:2])
            year = int(match[3][-2:])
            low, high = operation["year_range"]
            if (1 <= month <= 12 and 1 <= day <= 12
                    and (low <= year <= high or year == operation.get("boundary_year") and month > 1)):
                return f"{day}/{year}/{month:02d}"
        return None
    if kind == "numeric_display":
        text = "".join(value.split()) if operation.get("compact_input") else value
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        if not match:
            return None
        number = float(match[0])
        if not math.isfinite(number):
            return None
        fraction = operation.get("fraction")
        if fraction and (number > 1.0 or fraction != "above_one" and "%" in value):
            number /= 100
        low, high = operation["bounds"]
        if not low <= number <= high or (not number and operation.get("positive")):
            return None
        display = f"{number:.{operation['precision']}f}"
        if operation.get("compact") and "." in display:
            display = display.rstrip("0").rstrip(".")
        return display + operation.get("suffix", "")
    raise ValueError(f"Unknown direct repair operator: {kind}")


def direct_repairs(index, operations, allowed=()):
    scoped = {}
    for cell in allowed:
        scoped.setdefault(cell.column, []).append(cell.row_id)
    result = {}
    for operation in operations:
        column = operation["column"]
        field = index.fields[column]
        if not field.focus:
            raise ValueError(f"Direct repair targets context-only column: {column}")
        column_index = index.table.index[column]
        input_index = index.table.index[operation.get("from_column", column)]
        row_ids = scoped.get(column, ()) if operation.get("scope") == "allowed" else range(len(index.table.rows))
        for row_id in row_ids:
            row = index.table.rows[row_id]
            cell = Cell(row_id, column)
            raw = row[column_index]
            value = replacement(operation, row[input_index])
            if value is not None and value != raw.strip() and not field.issues(value):
                result.setdefault(cell, (value, operation["id"], operation["confidence"]))
    return result
