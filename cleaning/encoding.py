"""Deterministic categorical schema encoding from Dirty-only vocabularies."""

from collections import Counter
from .context import date_key


BUCKETS = {
    "pattern_bucket": ["pattern_match", "pattern_mismatch", "unknown"],
    "rarity_bucket": ["very_common", "common", "singleton", "very_rare", "rare", "mid_frequency"],
    "neighbor_bucket": ["no_neighbor_signal", "strong_support_other", "medium_support_other", "weak_support_other",
                        "strong_support_current", "weak_support_current"],
    "canonical_bucket": ["no_canonical_signal", "single_canonical_signal", "multi_canonical_signal"],
    "date_value_bucket": ["unparsed_date", "very_old", "old", "mid_old", "modern", "recent"],
    "candidate_rule_dominant": ["yes", "no"],
}


def context_categories(cell, summary, hits, settings, aliases):
    order = {kind: i for i, group in enumerate(settings.get("rule_order", [])) for kind in group}
    legacy = sorted((hit.rule.parameters["legacy"] for hit in hits if "legacy" in hit.rule.parameters),
                    key=lambda item: (order.get(item["rule_type"], len(order)), item.get("order", 0)))
    types = Counter(item["rule_type"] for item in legacy if item["rule_type"])
    roles = Counter(item["usage_role"] for item in legacy if item["usage_role"])
    rank, frequency = summary["value_frequency_rank"], summary["value_frequency"]
    rarity = ("very_common" if rank is not None and rank <= 3 else "common" if rank is not None and rank <= 10
              else "singleton" if frequency <= 1 else "very_rare" if frequency <= 2 else "rare" if frequency <= 5
              else "mid_frequency")
    ratio, current, majority = summary["neighbor_majority_ratio"], summary["value"], summary["neighbor_majority_value"]
    if ratio is None:
        neighbor = "no_neighbor_signal"
    elif str(majority) == str(current):
        neighbor = "strong_support_current" if ratio >= 0.8 else "weak_support_current"
    else:
        neighbor = "strong_support_other" if ratio >= 0.8 else "medium_support_other" if ratio >= 0.5 else "weak_support_other"
    count = summary["canonical_rule_count"]
    canonical = "no_canonical_signal" if count <= 0 else "single_canonical_signal" if count == 1 else "multi_canonical_signal"
    date = date_key(current) if summary["semantic_type"] == "date_like" else None
    year = int(date[:4]) if date else None
    date_bucket = ("unparsed_date" if year is None else "very_old" if year < 1950 else "old" if year < 1980
                   else "mid_old" if year < 2000 else "modern" if year < 2020 else "recent")
    language = (aliases.get(str(current).strip().lower()) if cell.column in settings.get("language_columns", []) else None)
    return {"column": cell.column, "semantic_type": summary["semantic_type"],
            "main_rule_type": types.most_common(1)[0][0] if types else "none",
            "main_usage_role": roles.most_common(1)[0][0] if roles else "none",
            "pattern_bucket": "pattern_match" if str(summary["current_pattern"]) == str(summary["dominant_pattern"]) else "pattern_mismatch",
            "rarity_bucket": rarity, "neighbor_bucket": neighbor, "canonical_bucket": canonical,
            "date_value_bucket": date_bucket, "language_bucket": f"lang_{language}" if language else "unknown_language",
            "candidate_rule_dominant": "yes" if summary["candidate_generation_rule_count"] > summary["strong_rule_count"] else "no"}


class CategoricalEncoding:
    def __init__(self, vocabularies):
        self.vocabularies = {name: tuple(sorted(set(values))) for name, values in vocabularies.items()}
        if any(not values or any(not isinstance(value, str) for value in values)
               for values in self.vocabularies.values()):
            raise ValueError("Categorical vocabularies need nonempty string categories")
        self.names = tuple(f"category:{name}={value}" for name, values in self.vocabularies.items() for value in values)

    def vector(self, values):
        # Unseen categories are all zero, matching handle_unknown='ignore'.
        return [float(values.get(name) == category) for name, categories in self.vocabularies.items()
                for category in categories]

    def snapshot(self):
        return {"vocabularies": self.vocabularies, "features": self.names,
                "source": "Dirty/schema only; categories absent from training are masked by Student"}
