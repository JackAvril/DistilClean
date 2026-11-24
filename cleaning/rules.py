"""Compile declarative and imported rules against a single dirty snapshot."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import groupby

from .data import shape


ARCHIVED_OPERATORS = {
    "actual_time_global_candidate": {"kind": "scan"},
    "ounces_canonical_format": {"kind": "canonical_number", "precision": 6},
    "abv_canonical_format": {"kind": "canonical_number", "precision": 15},
    "date_ymd_swap_suspicious": {"kind": "date_rotation", "pivot_year": 30},
}


def run_pattern(value):
    def classify(char):
        if char.isdigit():
            return "D"
        if char.isalpha():
            return "L"
        if char in " -_/.,()%:;&'+[]{}\"":
            return "S"
        return "X"

    return "".join(f"{kind}{sum(1 for _ in group)}" for kind, group in groupby(value.strip().lower(), classify))


def profile_rule(item, policy):
    rule_type = item["rule_type"]
    if rule_type == "rare_value":
        return {"kind": "listed_value", "values": item["rare_values"]}
    if rule_type == "rare_pattern":
        return {"kind": "run_pattern", "values": item["rare_patterns"], "deny": True}
    if rule_type == "pattern" and item.get("pattern"):
        return {"kind": "run_pattern", "values": [item["pattern"]]}
    if rule_type == "global_dominant_value":
        return {"kind": "global_expected", "value": item["dominant_value"],
                "support": item.get("dominant_ratio", 0)}
    if rule_type == "city_state_suffix_pollution":
        return {"kind": "strip_suffix", "pattern": r"\s+[a-z]{2}$"}
    if rule_type == "language_canonical_format":
        return {"kind": "canonical_alias", "aliases": policy["language_aliases"]}
    if rule_type == "author_list_braced_format":
        return {"kind": "regex", "pattern": '(?s)\\{[^"]*".*\\}'}
    if rule_type == "author_list_encoding_corruption":
        return {"kind": "contains", "tokens": ["\ufffd"]}
    if rule_type == "volume_issue_numeric_format":
        return {"kind": "regex_search", "pattern": r"\d+"}
    if rule_type == "date_like_format":
        return {"kind": "regex", "pattern": r"\s*(?:(?:0?[1-9]|1[0-2])\s*/\s*(?:0?[1-9]|[12]\d|3[01])\s*/\s*\d{2,4}|\d{4}\s*-\s*(?:0?[1-9]|1[0-2])\s*-\s*(?:0?[1-9]|[12]\d|3[01]))\s*"}
    if rule_type == "date_component_permutation_suspicious":
        return {"kind": "date_permutation", "distribution": item["date_distribution"]}
    if rule_type == "issn_canonical_format":
        return {"kind": "regex_search", "pattern": r"\d{4}-\d{3}[\dx]"}
    if rule_type in {"issn_format", "issn_loose_format", "pagination_format"}:
        return {"kind": "regex", "pattern": item["regex"],
                "allow_pattern": policy.get("regex_exceptions", {}).get(item["column"])}
    return {}


@dataclass
class Rule:
    id: str
    kind: str
    target: str
    weight: float
    parameters: dict
    triggers: bool = True


def candidate_trigger(item):
    flag = item.get("candidate_trigger")
    if isinstance(flag, str):
        enabled = flag.strip().lower() not in {"disabled", "disabled_evidence_only", "evidence_only", "false", "0"}
    else:
        enabled = flag is None or bool(flag)
    return (enabled and item.get("allow_candidate_generation") is not False
            and str(item.get("usage_role", "")).strip().lower() != "evidence_only")


def load_pool(payload, policy=None):
    """Translate supported legacy rule shapes and list the rest."""
    policy = policy or {}
    groups = payload.get("rules") if isinstance(payload, dict) else None
    if not isinstance(groups, dict) or not groups or any(not isinstance(items, list) for items in groups.values()):
        raise ValueError("Expected a grouped legacy rule pool")
    converted, skipped = [], []
    source_order = 0
    for rows in groups.values():
        for item in rows:
            source_order += 1
            rule_type = item["rule_type"]
            if rule_type in policy.get("disabled_rule_types", []):
                skipped.append({"id": item.get("rule_id"), "type": rule_type,
                                "reason": "disabled_by_import_policy"})
                continue
            column = item.get("rhs") or item.get("target_column") or item.get("column")
            archive_profile = policy.get("archived") and policy.get("profile_rules")
            if archive_profile and rule_type == "paired_missing_consistency" and len(item.get("columns", [])) == 2:
                first, second = item["columns"]
                common = {"kind": "paired_missing", "weight": 0.35, "triggers": candidate_trigger(item),
                          "archive_candidate": policy.get("candidate_scope") == "pool", "legacy_id": item["rule_id"]}
                common["legacy"] = {key: item.get(key) for key in
                                    ("rule_id", "rule_type", "priority", "usage_role", "support", "confidence")}
                common["legacy"]["order"] = source_order
                converted.extend({**common, "target": target, "other": other,
                                  "id": item["rule_id"] + ":" + target}
                                 for target, other in ((first, second), (second, first)))
                continue
            params = profile_rule(item, policy) if archive_profile else {}
            if policy.get("archived") and rule_type == "relationship_context_dominant":
                # The original predicate uses the stored map, literal trimmed values,
                # and an observed-value guard; it does not re-mine an FD.
                params = {"kind": "context", "lhs": item["lhs"], "majority_map": item["majority_map"],
                          "comparison": "literal", "observed_values": item.get("only_trigger_observed_values", []),
                          "missing_values": policy.get("predicate_missing", [])}
            elif policy.get("archived") and rule_type == "sample_patients_suffix":
                # canonical_suffix/max_edit_distance are mining metadata. The
                # archived checker actually reads required_suffix, defaulting to patients.
                params = {"kind": "suffix", "suffix": item.get("required_suffix", "patients").strip().lower(),
                          "missing_values": policy.get("predicate_missing", [])}
            if not params and rule_type in {"dominant_value_by_context", "dominant_value_by_context_pair"}:
                lhs = item.get("lhs") or item.get("context_column")
                lhs = [lhs] if isinstance(lhs, str) else lhs
                if lhs and "majority_map" in item:
                    params = {"kind": "context", "lhs": lhs, "majority_map": item["majority_map"]}
                elif lhs and "mappings" in item and len(lhs) == 1:
                    params = {"kind": "context", "lhs": lhs, "majority_map": {
                        row["context_value"]: {"dominant_value": row["dominant_value"], "ratio": row["dominant_ratio"]}
                        for row in item["mappings"]
                    }}
            elif not params and rule_type in {"functional_dependency", "soft_functional_dependency"} and item.get("lhs") and column:
                lhs = item["lhs"]
                params = {"kind": "fd", "lhs": [lhs] if isinstance(lhs, str) else lhs,
                          "confidence": item.get("confidence", 0.9), "min_support": 3,
                          "skip_missing_rhs": True}
                if policy.get("fd_group_policy") == "observed_majority":
                    # The archived confidence describes rule mining over the whole table.
                    params.update(confidence=0, min_support=1, allow_ties=True)
                limits = item.get("candidate_trigger_conditions", {})
                if item.get("candidate_trigger") == "strict_fd_only":
                    params.update(confidence=limits.get("min_dominant_ratio", 0.98),
                                  min_support=limits.get("min_group_size", 10),
                                  min_margin=limits.get("min_dominant_margin", 5), allow_ties=False)
            elif policy.get("archived") and rule_type in ARCHIVED_OPERATORS:
                params = dict(ARCHIVED_OPERATORS[rule_type])
            elif rule_type in {"not_null", "high_risk_missing"}:
                params = {"kind": "required"}
            elif rule_type == "enumeration_domain" or rule_type.endswith("_domain"):
                domain = item.get("valid_values") or item.get("allowed_values")
                if domain:
                    params = {"kind": "domain", "values": domain}
            elif not params and rule_type in {"pattern", "score_format", "sample_format", "state_abbr_format",
                               "author_list_braced_format"} and item.get("regex"):
                params = {"kind": "regex", "pattern": item["regex"]}
            if not params or not column:
                skipped.append({"id": item.get("rule_id"), "type": rule_type})
                continue
            params.update(
                target=column, id=item["rule_id"],
                weight={"high": 1.0, "medium": 0.65, "low": 0.35}.get(item.get("priority"), 0.65),
                triggers=candidate_trigger(item)
                and (rule_type != "soft_functional_dependency" or policy.get("soft_fd_triggers", False)),
            )
            if policy.get("archived") and policy.get("candidate_scope") == "pool":
                params["archive_candidate"] = True
            if policy.get("archived"):
                params["legacy"] = {key: item.get(key) for key in
                                    ("rule_id", "rule_type", "priority", "usage_role", "support", "confidence")}
                params["legacy"]["order"] = source_order
            converted.append(params)
    return converted, skipped


def suggested_value(rule, value):
    """Evaluate legacy format conflicts using the dirty string and declared convention."""
    params = rule.parameters
    if rule.kind == "canonical_number":
        matched = re.search(r"\d+(?:\.\d+)?", value)
        if not matched:
            return ""
        numeric = float(matched[0])
        if not math.isfinite(numeric):
            return ""
        precision = params["precision"]
        expected = f"{numeric:.{precision}f}".rstrip("0").rstrip(".")
        return expected if expected != value.strip().casefold() else ""
    matched = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{2,4})", "".join(value.split()))
    if not matched:
        return ""
    a, b, c = map(int, matched.groups())

    if rule.kind == "date_permutation":
        if not (1 <= a <= 12 and 1 <= b <= 31):
            return ""

        def expand_year(year):
            return year + (1900 if year >= 30 else 2000)

        current_year = expand_year(c) if len(matched[3]) == 2 else c
        distribution = params["distribution"]
        years = distribution.get("top_years", {})
        days = distribution.get("top_month_day", {})

        def score(month, day, year):
            return years.get(f"{year:04d}", 0) + days.get(f"{month:02d}-{day:02d}", 0)

        current = (a, b, current_year)
        baseline = score(*current)
        result, best = "", baseline
        for month, day, year in ((a, c, b), (b, c, a), (b, a, c)):
            canonical_year = expand_year(year) if 10 <= year <= 99 else year
            if not (1 <= month <= 12 and 1 <= day <= 31) or (month, day, canonical_year) == current:
                continue
            candidate_score = score(month, day, canonical_year)
            if candidate_score > best:
                result, best = f"{canonical_year:04d}-{month:02d}-{day:02d}", candidate_score
        return result if (best >= baseline + 2 or a == 1 or b == 1) else ""
    if not (1 <= a <= 12 and 1 <= b <= 12 and 1 <= c <= 31):
        return ""

    def year(raw):
        return raw + (1900 if raw >= params["pivot_year"] else 2000)

    current_year = year(c) if len(matched[3]) == 2 else c
    expected_year = year(a)
    if (a, b, current_year) == (b, c, expected_year):
        return ""
    # The archived high-recall rule treats bounded date components as a suspicious shape.
    return f"{b}/{c}/{expected_year % 100:02d}"


class RulePool:
    """Compile normalized rule specs and expose the fitted state to both stages."""

    def __init__(self, table, fields, specs):
        self.table = table
        self.fields = fields
        self.keys = [tuple(fields[col].key(value) for col, value in zip(table.columns, row))
                     for row in table.rows]
        # This historical denominator includes all nonblank shapes, even configured missing markers.
        self.patterns = {col: Counter(shape(row[j].strip()) for row in table.rows if row[j].strip())
                         for j, col in enumerate(table.columns)}
        self.by_target = defaultdict(list)
        self.relations = []
        self.rules = []
        self.specs = list(specs)
        ids = set()
        for spec in self.specs:
            rule = self.compile_rule(spec)
            if rule.id in ids:
                raise ValueError(f"Duplicate rule id: {rule.id}")
            ids.add(rule.id)
            self.by_target[rule.target].append(rule)
            self.rules.append(rule)
            if rule.kind == "relation":
                self.relations.append(rule)


    def compile_rule(self, spec):
        table = self.table
        kind, target = spec["kind"], spec["target"]
        table.index[target]
        parameters = dict(spec)
        if kind == "fd":
            lhs = tuple(spec["lhs"])
            if not lhs or target in lhs or len(set(lhs)) != len(lhs):
                raise ValueError(f"Invalid FD: {spec}")
            left = tuple(table.index[col] for col in lhs)
            right = table.index[target]
            groups = defaultdict(Counter)
            sources = spec.get("source")
            weights = spec.get("source_weights", {})
            source_index = table.index[sources] if sources else None
            for row in self.keys:
                key = tuple(row[j] for j in left)
                current = row[right]
                if all(key) and current:
                    weight = float(weights.get(row[source_index], 1)) if sources else 1.0
                    if not math.isfinite(weight) or weight < 0:
                        raise ValueError(f"Invalid vote weight for {spec['id']}")
                    if weight:
                        groups[key][current] += weight
            minimum = spec.get("min_support", 3)
            confidence = spec.get("confidence", 0.85)
            mapping = {}
            for key, counts in groups.items():
                # Counter preserves first observation for historical majority ties.
                allow_ties = spec.get("allow_ties", False)
                pair = counts.most_common(2)
                winner, votes = pair[0]
                total = sum(counts.values())
                runner_up = pair[1][1] if len(pair) > 1 else 0
                if (total >= minimum and votes / total >= confidence and (allow_ties or votes > runner_up)
                        and votes - runner_up >= spec.get("min_margin", 0)):
                    mapping[key] = (winner, votes / total)
            parameters.update(indices=left, mapping=mapping)
        elif kind == "context":
            lhs = tuple(spec["lhs"])
            if not lhs or target in lhs or len(set(lhs)) != len(lhs):
                raise ValueError(f"Invalid context key: {spec['id']}")
            mapping = {}
            for key, info in spec["majority_map"].items():
                parts = [key] if len(lhs) == 1 else key.split("||")
                if len(parts) != len(lhs) or info.get("dominant_value") is None:
                    raise ValueError(f"Invalid archived context entry: {spec['id']}")
                normalized = tuple(str(value).strip() if spec.get("comparison") == "literal"
                                   else self.fields[col].key(value) for col, value in zip(lhs, parts))
                if not all(normalized):
                    continue
                expected = str(info["dominant_value"])
                support = float(info["ratio"])
                if not 0 <= support <= 1:
                    raise ValueError(f"Invalid archived context confidence: {spec['id']}")
                previous = mapping.get(normalized)
                if previous is not None and previous != (expected, support):
                    raise ValueError(f"Context keys collide after normalization: {spec['id']}")
                mapping[normalized] = (expected, support)
            parameters.update(indices=tuple(table.index[col] for col in lhs), mapping=mapping)
        elif kind == "profile_pattern":
            minimum = spec.get("min_support", 20)
            confidence = spec.get("confidence", 0.95)
            if (type(minimum) is not int or minimum < 1
                    or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1):
                raise ValueError(f"Invalid profile thresholds for {spec['id']}")
            j = table.index[target]
            profile = Counter(shape(row[j].strip()) for row, keys in zip(table.rows, self.keys) if keys[j])
            eligible = sum(profile.values())
            dominant, count = min(profile.items(), key=lambda item: (-item[1], item[0]), default=("", 0))
            share = count / eligible if eligible else 0
            # A tied shape has no empirical winner. The confidence denominator excludes missing values.
            tied = sum(freq == count for freq in profile.values()) > 1
            parameters.update(dominant=dominant if count >= minimum and share >= confidence and not tied else "",
                              pattern_support=count, pattern_confidence=share)
        elif kind == "relation":
            if spec["relation"] not in {"difference", "reverse_difference", "circular_difference", "requires", "forbids"}:
                raise ValueError(f"Unknown relation: {spec['relation']}")
            parameters["other_index"] = table.index[spec["other"]]
            parameters["target_values"] = set(value.casefold() for value in spec.get("target_values", []))
            parameters["other_values"] = set(value.casefold() for value in spec.get("other_values", []))
        elif kind == "paired_missing":
            parameters["other_index"] = table.index[spec["other"]]
        elif kind in {"regex", "regex_search", "strip_suffix"}:
            parameters["compiled"] = re.compile(spec["pattern"], re.I)
            if spec.get("allow_pattern"):
                parameters["allow"] = re.compile(spec["allow_pattern"], re.I)
        elif kind in {"domain", "listed_value"}:
            parameters["keys"] = set(self.fields[target].key(value) for value in spec["values"])
        elif kind not in {"required", "scan", "canonical_number", "date_rotation", "run_pattern",
                          "global_expected", "canonical_alias", "contains", "date_permutation", "suffix"}:
            raise ValueError(f"Unsupported rule operator: {kind}")
        weight = float(spec.get("weight", 1))
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"Invalid weight for {spec['id']}")
        return Rule(spec["id"], kind, target, weight, parameters, spec.get("triggers", True))
