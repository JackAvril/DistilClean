from __future__ import annotations

import bisect
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache

from .data import Cell, Field, Table, shape
from .context import TypedContext
from .missing import MissingSemantics
from .encoding import CategoricalEncoding, BUCKETS, context_categories
from .contracts import CanonicalContracts, WeakEvidenceGuards
from .rules import Rule, RulePool, load_pool, run_pattern, suggested_value


FEATURE_NAMES = (
    "missing", "invalid", "value_frequency", "rarity", "pattern_frequency",
    "conflict_count", "conflict_weight", "expected_value_support",
    "neighbor_coverage", "heuristic_risk", "fd_conflict", "relation_conflict",
)


def sigmoid(value):
    return 1 / (1 + math.exp(-max(-40, min(40, value))))


@dataclass
class Signal:
    rule: Rule
    expected: str = ""
    support: float = 0.0


@dataclass
class Evidence:
    cell: Cell
    signals: list
    neighbors: Counter
    neighbor_count: int
    frequency: float
    rarity: float
    pattern_ratio: float
    probability: float
    missing: bool
    invalid: bool
    neighbor_examples: tuple
    context_summary: dict | None = None
    model_vector: list | None = None

    def vector(self):
        if self.model_vector is not None:
            return self.model_vector
        support = max((hit.support for hit in self.signals), default=0)
        strength = sum(hit.rule.weight for hit in self.signals)
        count = len(self.signals)
        if self.context_summary is not None:
            count = self.context_summary.get("violation_count", count)
            strength = self.context_summary.get("conflict_score", strength)
        return [
            float(self.missing), float(self.invalid), self.frequency, self.rarity,
            self.pattern_ratio, min(1, count / 5), min(1, strength / 4),
            support, min(1, self.neighbor_count / 8), self.probability,
            float(any(hit.rule.kind == "fd" for hit in self.signals)),
            float(any(hit.rule.kind == "relation" for hit in self.signals)),
        ]


class EvidenceIndex:
    """Dirty-table statistics, compiled rules and bounded row-neighbor lookup."""

    def __init__(self, table: Table, fields: dict[str, Field], config: dict, extra_rules=()):
        self.table = table
        self.fields = fields
        self.settings = config["detection"]
        self.contracts = CanonicalContracts(table, fields, config)
        self.valid_value_guards = WeakEvidenceGuards(fields, self.settings.get("valid_value_guards", []))
        context = self.settings.get("context", {})
        self.typed_context = TypedContext(table, context, self.settings["neighbors"]) if context else None
        self.numeric_feature_names = tuple(context.get("numeric_features", FEATURE_NAMES))
        categories = context.get("categorical_features", [])
        supported = {"column", "semantic_type", "main_rule_type", "main_usage_role", "language_bucket", *BUCKETS}
        if len(set(categories)) != len(categories) or set(categories) - supported:
            raise ValueError("Unsupported or repeated detector categorical features")
        self.pool = RulePool(table, fields, [*config.get("rules", []), *extra_rules])
        self.language_aliases = config.get("legacy_rule_import", {}).get("language_aliases", {})
        vocabularies = {}
        for name in categories:
            if name == "column":
                vocabularies[name] = [col for col, field in fields.items() if field.focus]
            elif name == "semantic_type":
                if self.typed_context is None:
                    raise ValueError("Semantic categories require typed context")
                vocabularies[name] = [profile.semantic_type for profile in self.typed_context.columns.values()]
            elif name in BUCKETS:
                vocabularies[name] = BUCKETS[name]
            elif name == "language_bucket":
                vocabularies[name] = ["unknown_language", *(f"lang_{value}" for value in self.language_aliases.values())]
            else:
                key = "rule_type" if name == "main_rule_type" else "usage_role"
                vocabularies[name] = ["none", *(rule.parameters["legacy"][key] for rule in self.pool.rules
                                               if rule.parameters.get("legacy", {}).get(key))]
        self.encoding = CategoricalEncoding(vocabularies)
        self.feature_names = (*self.numeric_feature_names, *self.encoding.names)
        self.missing = MissingSemantics(table, fields, config)
        self.keys = self.pool.keys
        self.patterns = self.pool.patterns
        self.counts = {}
        self.postings = {}
        self.display = {}
        self.frequent = {}
        for j, col in enumerate(table.columns):
            counts = Counter(row[j] for row in self.keys)
            self.counts[col] = counts
            postings, display = defaultdict(list), defaultdict(Counter)
            for i, row in enumerate(self.keys):
                if row[j]:
                    postings[row[j]].append(i)
                    display[row[j]][table.rows[i][j]] += 1
            self.postings[col] = postings
            self.display[col] = {key: min(values, key=lambda value: (-values[value], value))
                                 for key, values in display.items()}
            self.frequent[col] = sorted(display, key=lambda key: (-counts[key], key))[:32]
        self.by_target = self.pool.by_target
        self.relations = self.pool.relations
        self.rules = self.pool.rules
        self.archive_scope = any(rule.parameters.get("archive_candidate") for rule in self.rules)

    def relation_violation(self, rule, value, other):
        p = rule.parameters
        field, other_field = self.fields[rule.target], self.fields[p["other"]]
        if p.get("require_valid_other") and other_field.issues(other):
            return False
        if p["relation"] in {"difference", "reverse_difference", "circular_difference"}:
            left, right = field.numeric(value), other_field.numeric(other)
            if left is None or right is None:
                return False
            difference = left - right
            if p["relation"] == "reverse_difference":
                difference = -difference
            if p["relation"] == "circular_difference":
                difference = (difference + 720) % 1440 - 720
            return not p["min"] <= difference <= p["max"]
        key, other_key = field.key(value), other_field.key(other)
        if key not in p["target_values"] or not other_key:
            return False
        match = other_key in p["other_values"]
        return match if p["relation"] == "forbids" else not match

    def schema_issues(self, cell, value):
        field = self.fields[cell.column]
        issues = field.issues(value)
        if not field.key(value):
            required = self.missing.requirement(cell)
            if required is False:
                issues = tuple(issue for issue in issues if issue != "required")
            elif required is True and "required" not in issues:
                issues = (*issues, "required")
        return issues

    def check(self, cell, value):
        col = cell.column
        field = self.fields[col]
        key = field.key(value)
        issues = self.schema_issues(cell, value)
        signals = [Signal(Rule(f"{col}:{issue}", "schema", col, 1.0, {})) for issue in issues]
        row = self.keys[cell.row_id]
        for rule in self.by_target[col]:
            if not key and self.missing.suppresses(cell, rule):
                continue
            p = rule.parameters
            if rule.kind == "fd":
                expected, support = p["mapping"].get(tuple(row[j] for j in p["indices"]), ("", 0))
                if expected and key != expected and (key or not p.get("skip_missing_rhs")):
                    signals.append(Signal(rule, self.display[col][expected], support))
            elif rule.kind == "context":
                if p.get("comparison") == "literal":
                    observed = value.strip()
                    context_key = tuple(self.table.rows[cell.row_id][j].strip() for j in p["indices"])
                    missing = p.get("missing_values", [])
                    if (observed.lower() in missing or any(part.lower() in missing for part in context_key)
                            or p.get("observed_values") and observed not in p["observed_values"]):
                        continue
                    expected, support = p["mapping"].get(context_key, ("", 0))
                    mismatch = bool(expected) and observed != expected
                else:
                    expected, support = p["mapping"].get(tuple(row[j] for j in p["indices"]), ("", 0))
                    mismatch = key and expected and key != field.key(expected)
                if mismatch:
                    signals.append(Signal(rule, expected, support))
            elif rule.kind == "suffix":
                observed = value.strip().lower()
                if observed not in p.get("missing_values", []) and not observed.endswith(p["suffix"]):
                    signals.append(Signal(rule))
            elif rule.kind == "relation":
                other_value = self.table.rows[cell.row_id][p["other_index"]]
                if self.relation_violation(rule, value, other_value):
                    signals.append(Signal(rule))
            elif rule.kind == "paired_missing":
                if not key and row[p["other_index"]]:
                    signals.append(Signal(rule))
            elif rule.kind in {"canonical_number", "date_rotation", "date_permutation"} and key:
                expected = suggested_value(rule, value)
                if expected:
                    signals.append(Signal(rule, expected))
            elif rule.kind == "strip_suffix" and key:
                observed = value.strip().lower()
                expected = p["compiled"].sub("", observed).strip()
                if expected != observed:
                    signals.append(Signal(rule, expected))
            elif key and (rule.kind == "listed_value" and key in p["keys"]
                          or rule.kind == "run_pattern" and (run_pattern(value) in p["values"]) == p.get("deny", False)
                          or rule.kind == "contains" and any(token in value for token in p["tokens"])
                          or rule.kind == "regex_search" and not p["compiled"].search(value.strip())):
                signals.append(Signal(rule))
            elif key and rule.kind == "global_expected" and key != field.key(p["value"]):
                signals.append(Signal(rule, p["value"], p["support"]))
            elif key and rule.kind == "canonical_alias" and key != p["aliases"].get(key):
                signals.append(Signal(rule, p["aliases"].get(key, "")))
            elif (rule.kind == "required" and not key
                  or rule.kind == "regex" and key and not p["compiled"].fullmatch(value.strip())
                  and not (p.get("allow") and p["allow"].fullmatch(value.strip()))
                  or rule.kind == "domain" and key and key not in p["keys"]
                  or rule.kind == "profile_pattern" and key and p["dominant"]
                  and shape(value.strip()) != p["dominant"]):
                signals.append(Signal(rule))
        return signals

    def candidate_cells(self, row_ids=None):
        for rid in range(len(self.table.rows)) if row_ids is None else row_ids:
            row = self.table.rows[rid]
            for col, value in zip(self.table.columns, row):
                field = self.fields[col]
                if not field.focus:
                    continue
                cell = Cell(rid, col)
                hits = self.check(cell, value)
                hits.extend(Signal(rule) for rule in self.by_target[col] if rule.kind == "scan")
                if self.archive_scope:
                    if any(hit.rule.triggers and hit.rule.parameters.get("archive_candidate") for hit in hits):
                        yield cell, hits
                    elif cell in self.contracts.matches:
                        match = self.contracts.matches[cell]
                        hits.append(Signal(Rule(f"contract:{match.operation_id}", "canonical_contract", col,
                                                match.confidence, {}), match.expected, 1.0))
                        yield cell, hits
                    continue
                key = self.keys[rid][self.table.index[col]]
                count = self.counts[col].get(key, 0)
                profile = self.patterns[col]
                rare = key and count <= self.settings["rare_max_count"]
                unusual = profile and profile.get(shape(value.strip()), 0) / len(self.keys) <= self.settings["rare_pattern_ratio"]
                if rare and (unusual or field.kind == "category"):
                    hits.append(Signal(Rule(f"{col}:rare", "rarity", col, 0.35, {})))
                if any(hit.rule.triggers for hit in hits):
                    yield cell, hits
                elif cell in self.contracts.matches:
                    match = self.contracts.matches[cell]
                    hits.append(Signal(Rule(f"contract:{match.operation_id}", "canonical_contract", col,
                                            match.confidence, {}), match.expected, 1.0))
                    yield cell, hits

    def consensus_cells(self, confidence):
        for rule in self.rules:
            if rule.kind != "fd" or not rule.parameters.get("expand") or not self.fields[rule.target].focus:
                continue
            p = rule.parameters
            target = self.table.index[rule.target]
            for rid, row in enumerate(self.keys):
                expected, support = p["mapping"].get(tuple(row[j] for j in p["indices"]), ("", 0))
                if expected and support >= confidence and row[target] != expected:
                    yield Cell(rid, rule.target)

    @lru_cache(maxsize=8192)
    def neighbors(self, cell):
        if self.typed_context is not None:
            return self.typed_context.neighbors(cell)
        row = self.keys[cell.row_id]
        limit = self.settings["neighbor_pool"]
        columns = [(j, col, key, self.postings[col].get(key, ()))
                   for j, (col, key) in enumerate(zip(self.table.columns, row))
                   if col != cell.column and key and self.fields[col].focus]
        # Bound work even when the most common value occurs in nearly every row.
        columns.sort(key=lambda entry: (len(entry[3]), entry[0]))
        blocked = set()
        for _, _, _, postings in columns:
            i = bisect.bisect_left(postings, cell.row_id)
            per_key = max(2, limit // max(1, len(columns)))
            blocked.update(postings[max(0, i - per_key // 2):i + per_key // 2 + 1])
        blocked.discard(cell.row_id)
        candidates = []
        for rid in blocked:
            score = sum(self.keys[rid][j] == key for j, _, key, _ in columns)
            if score:
                candidates.append((-score, abs(rid - cell.row_id), rid))
        candidates.sort()
        top = candidates[:self.settings["neighbors"]]
        return tuple((rid, -score / len(columns)) for score, _, rid in top)

    def evidence(self, cell, hits=None):
        value = self.table.at(cell)
        field = self.fields[cell.column]
        key = field.key(value)
        hits = self.check(cell, value) if hits is None else hits
        neighbors = self.neighbors(cell)
        matches = Counter(self.keys[rid][self.table.index[cell.column]] for rid, _ in neighbors)
        matches.pop("", None)
        total = max(1, len(self.keys))
        frequency = self.counts[cell.column].get(key, 0) / total
        pattern_ratio = self.patterns[cell.column].get(shape(value.strip()), 0) / total
        disagreement = 1 - matches[key] / sum(matches.values()) if matches else 0
        rarity = min(1, -math.log(max(frequency, 1 / total)) / max(1, math.log(total))) if key else 0
        weight = sum(hit.rule.weight for hit in hits)
        logit = -2.2 + 1.45 * min(3, weight) + 0.65 * rarity + 0.8 * disagreement
        probability = sigmoid(logit)
        summary = self.typed_context.summary(cell, neighbors) if self.typed_context is not None else None
        if summary and self.typed_context.settings.get("statistics") == "typed":
            frequency = summary["value_frequency"] / total
            pattern_ratio = summary["pattern_count"] / total
            rarity = 1 - summary["value_frequency"] / max(1, summary["non_null_count"])
        if summary and "probability" in self.typed_context.settings:
            summary.update(self.context_probability(cell, hits, summary))
            probability = summary["posterior_error_probability"]
        model_vector = None
        if summary and "numeric_features" in self.typed_context.settings:
            summary.update(self.context_features(cell, hits, summary))
            model_vector = [float(summary[name] or 0) for name in self.numeric_feature_names]
        if self.encoding.names:
            if model_vector is None:
                raise ValueError("Categorical encoding requires configured numeric context features")
            categories = context_categories(cell, summary, hits, self.typed_context.settings, self.language_aliases)
            summary.update(categories)
            model_vector.extend(self.encoding.vector(categories))
        return Evidence(cell, hits, matches, len(neighbors), frequency, rarity, pattern_ratio,
                        probability, not key, bool(self.schema_issues(cell, value)), neighbors, summary, model_vector)

    def context_features(self, cell, hits, summary):
        settings = self.typed_context.settings
        legacy = [hit.rule.parameters["legacy"] for hit in hits if "legacy" in hit.rule.parameters]
        counts = {}
        for name, groups in settings["rule_features"].items():
            counts[name] = sum(signal["rule_type"] in groups.get("types", ()) or
                               signal["usage_role"] in groups.get("roles", ()) for signal in legacy)
        counts["schema_rule_count"] += counts["pattern_rule_count"]
        strong, candidate = counts["strong_rule_count"], counts["candidate_generation_rule_count"]
        total = strong + candidate
        ratio = summary["neighbor_majority_ratio"] or 0
        majority, value = summary["neighbor_majority_value"], summary["value"]
        compare = lambda item: "__NULL__" if item is None else str(item).strip()
        equal = compare(majority) == compare(value)
        numeric = 0
        if cell.column in settings["numeric_columns"] and value is not None:
            match = re.search(r"\d+", value)
            numeric = int(match[0]) if match else 0
        rank = summary["value_frequency_rank"]
        rank = settings.get("missing_rank", 999999) if rank is None else rank
        rare = rank > settings["rare_rank_threshold"]
        if settings.get("rarity_mode", "frequency_or_rank") == "frequency_or_rank":
            rare = rare or summary["value_frequency"] <= settings["rare_frequency_threshold"]
        strength_weights = settings.get("derived_strength_weights")
        strength = (sum(weight * counts[name] for name, weight in strength_weights.items())
                    if strength_weights else sum(counts[name] for name in settings["strength_features"]))
        disagree = (ratio > 0 and not equal if settings.get("disagreement_mode") == "unequal"
                    else majority is not None and value is not None and majority != value and ratio > 0)
        return {
            **counts, "numeric_value": numeric,
            "candidate_rule_ratio": candidate / (total + settings.get("candidate_ratio_epsilon", 0)) if total else 0,
            "rule_total_count": total, "strong_rule_ratio": strong/(total+1e-8) if total else 0,
            "context_rule_ratio": counts["context_rule_count"]/(total+1e-8) if total else 0,
            "fd_like_ratio": counts["fd_like_count"]/(total+1e-8) if total else 0,
            "global_rule_ratio": counts["global_rule_count"]/(total+1e-8) if total else 0,
            "rule_strength_sum": strength,
            "has_neighbor_signal": int(ratio > 0),
            "high_neighbor_consistency": int(ratio >= settings["neighbor_consistency_threshold"]),
            "is_rare_value": int(rare),
            "neighbor_disagree": int(disagree),
            "is_equal_to_neighbor_majority": int(equal),
            "neighbor_agreement_score": ratio if equal else -ratio,
        }

    @lru_cache(maxsize=1)
    def candidate_counts(self):
        return Counter(cell.column for cell, _ in self.candidate_cells())

    def context_probability(self, cell, hits, summary):
        settings = self.typed_context.settings["probability"]
        signals = [hit.rule.parameters["legacy"] for hit in hits if "legacy" in hit.rule.parameters]
        weight = 0
        for signal in signals:
            priority = settings["priority_weights"].get(signal["priority"], 0.6)
            kind = settings["type_weights"].get(signal["rule_type"], 1.0)
            support = max(int(signal["support"] or 0), 1)
            weight += round((0.1 + float(signal["confidence"] or 0)) * math.log1p(support) * priority * kind, 6)
        conflict = round(weight, 6)
        prior = min(0.95, max(0.0001, self.candidate_counts()[cell.column] / max(1, len(self.table.rows))))
        rule = min(5, len(signals) + conflict / 100)
        rarity = 1 - summary["value_frequency"] / max(1, summary["non_null_count"])
        rarity *= settings["rarity_weights"].get(summary["semantic_type"], 1.0)
        pattern = float(bool(summary["dominant_pattern"]) and
                        summary["current_pattern"] != summary["dominant_pattern"])
        majority, ratio = summary["neighbor_majority_value"], summary["neighbor_majority_ratio"]
        neighbor = (ratio if majority is not None and ratio is not None and majority != summary["value"] else 0)
        transform = settings.get("prior_logit", "clipped")
        if transform not in {"clipped", "complement_cap"}:
            raise ValueError(f"Unsupported context prior transform: {transform}")
        logit_prior = min(max(prior, 1e-6), 1-prior) if transform == "complement_cap" else prior
        logit = math.log(logit_prior / (1 - logit_prior)) + sum(a * b for a, b in
                   zip(settings["weights"], (rule, rarity, pattern, neighbor)))
        return {"violation_count": len(signals), "conflict_score": conflict,
                "prior_error_probability": round(prior, 6), "posterior_error_probability": round(sigmoid(logit), 6),
                "rule_signal": round(rule, 6), "rarity_signal": round(rarity, 6),
                "pattern_signal": pattern, "neighbor_signal": round(neighbor, 6)}

    def candidate_sort_key(self, item):
        groups = self.typed_context.settings["rule_order"]
        priorities = {kind: i for i, group in enumerate(groups) for kind in group}
        first = min((priorities[hit.rule.parameters["legacy"]["rule_type"]],
                     hit.rule.parameters["legacy"]["order"])
                    for hit in item.signals if "legacy" in hit.rule.parameters)
        return (-item.context_summary["conflict_score"], -item.context_summary["violation_count"],
                item.cell.row_id, *first)

    def context(self, evidence):
        cell = evidence.cell
        col = cell.column
        row = self.table.rows[cell.row_id]
        return {
            "column": col, "value": self.table.at(cell),
            "row": dict(zip(self.table.columns, row)),
            "column_top": [[self.display[col][key], self.counts[col][key]] for key in self.frequent[col][:5]],
            "conflicts": [{"rule": signal.rule.id, "expected": signal.expected, "support": signal.support}
                          for signal in evidence.signals[:12]],
            "neighbors": [{"value": self.table.at(Cell(row_id, col)), "similarity": score}
                          for row_id, score in evidence.neighbor_examples],
            "heuristic_risk": evidence.probability,
            **self.contracts.context(cell),
            **({"typed_context": evidence.context_summary} if evidence.context_summary is not None else {}),
        }
