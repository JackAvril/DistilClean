"""Typed dirty-table profiles and full-table row similarity with stable ties."""

import math
import re
from collections import Counter
from functools import lru_cache

import numpy as np
from scipy.sparse import csr_matrix

from .rules import run_pattern


def parse_number(value):
    if value is None:
        return None
    try:
        number = float(value.replace(",", "").replace("%", ""))
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def date_key(value):
    if value is None:
        return None
    compact = "".join(value.lower().split())
    match = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{2,4})", compact)
    if match:
        month, day, year = map(int, match.groups())
        if len(match[3]) == 2:
            year += 1900 if year >= 30 else 2000
    else:
        match = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", compact)
        if not match:
            return None
        year, month, day = map(int, match.groups())
    return f"{year:04d}-{month:02d}-{day:02d}" if 1 <= month <= 12 and 1 <= day <= 31 else None


class ColumnProfile:
    def __init__(self, values, semantic_type=None, thresholds=None):
        thresholds = thresholds or {}
        numeric_min = thresholds.get("numeric_ratio", 0.9)
        categorical_max = thresholds.get("categorical_unique_count", 50)
        categorical_ratio = thresholds.get("categorical_unique_ratio", 0.08)
        id_ratio = thresholds.get("id_unique_ratio", 0.98)
        code_length = thresholds.get("code_average_length", 20)
        self.values = np.array(values, dtype=object)
        self.present = np.array([value is not None for value in values])
        self.counts = Counter(value for value in values if value is not None)
        self.total = sum(self.counts.values())
        self.ranks = {value: i for i, (value, _) in enumerate(self.counts.most_common(), 1)}
        self.patterns = np.array([run_pattern(value) if value is not None else "NULL" for value in values])
        patterns = Counter(pattern for pattern, present in zip(self.patterns, self.present) if present)
        self.dominant_pattern = next(iter(patterns.most_common(1)), ("", 0))[0]
        self.pattern_counts = patterns
        self.numbers = np.array([parse_number(value) for value in values], dtype=float)
        numeric_ratio = np.isfinite(self.numbers).sum() / self.total if self.total else 0
        unique_ratio = len(self.counts) / self.total if self.total else 0
        avg_length = sum(len(value) * count for value, count in self.counts.items()) / self.total if self.total else 0
        dominant_ratio = patterns[self.dominant_pattern] / self.total if self.total else 0
        detected = "numeric" if numeric_ratio >= numeric_min else (
            "categorical" if len(self.counts) <= categorical_max or unique_ratio <= categorical_ratio else "text")
        id_like = self.total and unique_ratio >= id_ratio and (avg_length <= code_length or numeric_ratio >= numeric_min)
        code_like = self.total and (avg_length <= code_length and unique_ratio > categorical_ratio and dominant_ratio >= 0.7
                                   or avg_length <= 15 and numeric_ratio >= 0.95 and unique_ratio < id_ratio)
        self.semantic_type = semantic_type or ("id_like" if id_like else "code_like" if code_like else detected)
        if self.semantic_type not in {"numeric", "date_like", "id_like", "code_like", "categorical", "text"}:
            raise ValueError(f"Unsupported context semantic type: {self.semantic_type}")
        self.dates = np.array([date_key(value) for value in values], dtype=object)
        self.tokens = None
        if self.semantic_type == "text":
            vocabulary, indices, indptr = {}, [], [0]
            for value in values:
                for token in sorted(set(value.lower().split()) if value is not None else ()):
                    indices.append(vocabulary.setdefault(token, len(vocabulary)))
                indptr.append(len(indices))
            self.tokens = csr_matrix((np.ones(len(indices), dtype=np.int32), indices, indptr),
                                     shape=(len(values), len(vocabulary)))
            self.token_counts = np.diff(indptr)

    def similarities(self, row):
        present = self.present & self.present[row]
        comparable = self.present | self.present[row]
        same_pattern = (self.patterns == self.patterns[row]) & present
        exact = (self.values == self.values[row]) & present
        result = np.zeros(len(self.values), dtype=float)
        if self.semantic_type == "numeric":
            numbers, query = self.numbers, self.numbers[row]
            finite = np.isfinite(numbers) & np.isfinite(query) & present
            denominator = np.maximum(np.maximum(np.abs(numbers[finite]), abs(query)), 1.0)
            result[finite] = np.maximum(0, 1 - np.abs(numbers[finite] - query) / denominator)
        elif self.semantic_type == "date_like":
            valid = np.array([value is not None for value in self.dates])
            parsed = valid & valid[row] & present
            result[same_pattern & ~parsed] = 0.2
            result[parsed & (self.dates == self.dates[row])] = 1.0
        else:
            result[same_pattern] = 0.2
            if self.tokens is not None:
                overlap = (self.tokens @ self.tokens.getrow(row).T).toarray().ravel()
                union = self.token_counts + self.token_counts[row] - overlap
                jaccard = np.divide(overlap, union, out=np.ones(len(union), dtype=float), where=union != 0)
                result = np.maximum(result, np.where(present, jaccard, 0))
            result[exact] = 1.0
        return result, comparable.astype(np.int16)


class TypedContext:
    def __init__(self, table, settings, neighbors):
        self.table = table
        self.limit = neighbors
        self.settings = settings
        missing = set(settings["missing"])
        self.columns = {}
        for col, j in table.index.items():
            values = [value.strip() if value.strip().lower() not in missing else None
                      for value in (row[j] for row in table.rows)]
            self.columns[col] = ColumnProfile(values, settings.get("types", {}).get(col),
                                               settings.get("type_thresholds"))

    @lru_cache(maxsize=128)
    def row_similarities(self, row):
        # A query batch occupies O(columns * rows) memory, including on large tables.
        return {col: profile.similarities(row) for col, profile in self.columns.items()}

    def neighbors(self, cell):
        signals = self.row_similarities(cell.row_id)
        score = np.zeros(len(self.table.rows), dtype=float)
        count = np.zeros(len(self.table.rows), dtype=np.int16)
        for col in self.table.columns:
            if col != cell.column:
                values, present = signals[col]
                score += values
                count += present
        score = np.divide(score, count, out=np.zeros(len(score)), where=count != 0)
        # Historical ranking rounds before sorting and preserves input-row order on ties.
        rounded = [round(float(value), 6) for value in score]
        positions = sorted((i for i in range(len(score)) if i != cell.row_id), key=lambda i: -rounded[i])
        return tuple((i, rounded[i]) for i in positions[:self.limit])

    def summary(self, cell, neighbors):
        profile = self.columns[cell.column]
        observed = profile.values[cell.row_id]
        if observed is not None and self.settings.get("candidate_case") == "lower":
            observed = observed.lower()
        distribution = Counter(profile.values[rid] for rid, _ in neighbors if profile.values[rid] is not None)
        majority, count = next(iter(distribution.most_common(1)), (None, 0))
        pattern = run_pattern(observed) if observed is not None else "NULL"
        return {
            "semantic_type": profile.semantic_type, "value": observed,
            "value_frequency": profile.counts.get(observed, 0), "value_frequency_rank": profile.ranks.get(observed),
            "non_null_count": profile.total, "pattern_count": profile.pattern_counts.get(pattern, 0),
            "dominant_pattern": profile.dominant_pattern, "current_pattern": pattern,
            "neighbor_majority_value": majority,
            "neighbor_majority_ratio": round(count / len(neighbors), 6) if distribution and neighbors else None,
            "neighbor_value_distribution": dict(distribution),
        }
