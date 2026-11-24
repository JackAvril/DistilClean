"""Repair-time candidate evidence from independently identified sources."""

import math
import re
from collections import Counter, defaultdict

from .data import Cell
from .repair_rules import replacement


def context_candidates(index, relations, formats):
    """Index the observed alternatives per context, retaining more than the majority."""
    by_column = defaultdict(list)
    for relation in relations:
        column = relation["column"]
        field = index.fields[column]
        left = tuple(index.table.index[c] for c in relation["group"])
        right = index.table.index[column]
        groups = defaultdict(Counter)
        fmt = formats.get(column)
        for row, keys in zip(index.table.rows, index.keys):
            context = tuple(keys[i] for i in left)
            if not all(context) or not keys[right]:
                continue
            value = replacement(fmt, row[right]) if fmt else index.display[column][keys[right]]
            if value is not None and not field.issues(value):
                groups[context][value] += 1
        by_column[column].append((left, {key: [(value, count / sum(counter.values()))
                                             for value, count in counter.most_common(relation["top"])]
                                         for key, counter in groups.items()}))
    return by_column


def group_consensus(index, settings):
    """Count dirty rows in configured groups and suggest a sufficiently dominant value."""
    votes, expand = {}, set()
    normalization = settings.get("normalize", {})
    missing = set(settings.get("missing", []))

    def normalized(name, raw):
        f = index.fields[name]
        raw = raw.strip()
        if not f.key(raw) or f.key(raw) in missing:
            return ""
        rules = normalization.get(name, {})
        if "pattern" in rules:
            raw = re.sub(rules["pattern"], rules.get("replacement", ""), raw, flags=re.I)
        value = next(f.transforms(raw), raw) if rules.get("typed") else raw
        if rules.get("casefold"):
            value = value.casefold()
        return rules.get("aliases", {}).get(f.key(raw), value)

    for spec in settings.get("relations", []):
        columns, column = spec["group"], spec["column"]
        lhs = [index.table.index[c] for c in columns]
        rhs = index.table.index[column]
        field = index.fields[column]
        groups = defaultdict(Counter)
        positions = []
        for raw in index.table.rows:
            group = tuple(index.fields[c].key(normalized(c, raw[i])) for c, i in zip(columns, lhs))
            value = normalized(column, raw[rhs])
            positions.append((group, value))
            if all(group) and value:
                groups[group][value] += 1
        winners = {}
        for group, counter in groups.items():
            (value, best), *other = counter.most_common(2)
            total = sum(counter.values())
            confidence = best / total
            margin = (best - (other[0][1] if other else 0)) / total
            if best >= settings["min_support"] and confidence >= settings["confidence"] and margin >= settings["margin"]:
                winners[group] = value, confidence
        for row_id, (group, value) in enumerate(positions):
            winner = winners.get(group)
            if winner and field.key(value) != field.key(winner[0]):
                cell = Cell(row_id, column)
                # Multiple valid relations can target a cell; the later declaration wins.
                votes[cell] = winner
    expand = {cell for cell, (value, _) in votes.items() if cell.column in settings.get("expand", ())
              and index.fields[cell.column].focus and not index.fields[cell.column].issues(value)}
    return votes, expand


def source_consensus(index, specs):
    """Vote by configured reliability and require distinct corroborating sources."""
    result = {}
    for spec in specs:
        group_indices = [index.table.index[col] for col in spec["group"]]
        source_index = index.table.index[spec["source"]]
        weights = spec["weights"]
        if not group_indices or any(not math.isfinite(w) or w < 0 for w in weights.values()):
            raise ValueError("Invalid source consensus votes")
        for column in spec["columns"]:
            j = index.table.index[column]
            field = index.fields[column]
            groups = defaultdict(Counter)
            support = defaultdict(Counter)
            voters = defaultdict(set)
            row_groups = []
            for keys, raw in zip(index.keys, index.table.rows):
                group = tuple(keys[i] for i in group_indices)
                value = next(field.transforms(raw[j]), raw[j]) if field.kind == "time" else raw[j]
                row_groups.append((group, value))
                source = keys[source_index]
                weight = weights.get(source, 0)
                if not all(group) or not weight or not field.key(value) or field.issues(value):
                    continue
                groups[group][value] += weight
                support[group][value] += 1
                voters[(group,value)].add(source)
            winners = {}
            for group, votes in groups.items():
                value, weight = votes.most_common(1)[0]
                confidence = weight / sum(votes.values())
                if (support[group][value] >= spec["min_support"]
                        and len(voters[group,value]) >= spec["min_sources"]
                        and confidence >= spec["confidence"]):
                    winners[group] = value, confidence, len(voters[group,value])
            for row_id, (group, value) in enumerate(row_groups):
                winner = winners.get(group)
                if winner and field.key(value) != field.key(winner[0]):
                    cell = Cell(row_id, column)
                    if cell in result:
                        raise ValueError(f"Overlapping source consensus definitions for {column}")
                    result[cell] = winner
    return result
