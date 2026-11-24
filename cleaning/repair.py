from __future__ import annotations

import random
from dataclasses import dataclass
from difflib import SequenceMatcher

from .consensus import context_candidates, group_consensus, source_consensus
from .data import Cell, shape, time_minutes, write_csv, write_json
from .models import Student
from .repair_rules import direct_repairs, replacement
from .sampling import diverse_sample


FEATURES = [
    "keep_original", "edit_similarity", "length_similarity", "same_pattern",
    "value_frequency", "neighbor_agreement", "expected_value_support",
    "conflict_reduction", "remaining_conflict", "valid", "missing", "type_distance_available",
    "type_distance", "transform_source", "domain_source",
]


@dataclass
class Candidate:
    value: str
    sources: list[str]
    features: list[float]
    heuristic: float
    admissible: bool
    score: float = 0.0


@dataclass
class Repair:
    cell: Cell
    dirty_value: str
    proposed_value: str
    repaired_value: str
    score: float
    margin: float
    accepted: bool
    reason: str
    source: str


def typed_distance(field, old, new):
    left, right = field.numeric(old), field.numeric(new)
    if left is None or right is None:
        return False, 0
    delta = abs(left - right)
    if field.kind == "time":
        return True, min(delta, 1440 - delta) / 720
    scale = max(1, abs(left), abs(right))
    return True, abs(left / scale - right / scale)


def candidate_set(index, evidence, settings, consensus=None, group_vote=None, direct=None, contexts=()):
    cell = evidence.cell
    col, old = cell.column, index.table.at(cell)
    field = index.fields[col]
    available = {}

    def add(value, source):
        if value != old and (not field.key(value) or field.issues(value)):
            return
        sources = available.setdefault(value, [])
        if source not in sources:
            sources.append(source)

    add(old, "original")
    if direct:
        add(direct[0], "direct:" + direct[1])
    for value in field.transforms(old):
        add(value, "transform")
    rule = settings.get("candidate_formats", {}).get(col)
    if rule and field.key(old):
        formatted = replacement(rule, old)
        if formatted is not None and formatted != old:
            add(formatted, "transform")
    for hit in evidence.signals:
        if hit.expected:
            add(hit.expected, "rule")
    if consensus:
        add(consensus[0], "source_consensus")
    if group_vote:
        add(group_vote[0], "group_consensus")
    context_support = {}
    for indices, mapping in contexts:
        group = tuple(index.keys[cell.row_id][i] for i in indices)
        for value, support in mapping.get(group, ()):
            add(value, "context_dictionary")
            context_support[value] = max(support, context_support.get(value, 0))
    for key, _ in evidence.neighbors.most_common(8):
        add(index.display[col][key], "neighbor")
    for key in index.frequent[col][:settings["frequency_candidates"]]:
        add(index.display[col][key], "frequency")
    for key, display in field.domain.items():
        add(index.display[col].get(key, display), "domain")

    old_conflict = sum(hit.rule.weight for hit in evidence.signals if hit.rule.kind != "rarity")
    old_relations = {hit.rule.id for hit in evidence.signals if hit.rule.kind == "relation"}
    result = []
    for value, sources in available.items():
        key = field.key(value)
        hits = index.check(cell, value)
        strength = sum(hit.rule.weight for hit in hits)
        reduction = (old_conflict - strength) / max(1, old_conflict, strength)
        support = max((hit.support for hit in evidence.signals
                       if hit.expected and field.key(hit.expected) == key), default=0)
        if consensus and key == field.key(consensus[0]):
            support = max(support, consensus[1])
        if group_vote and key == field.key(group_vote[0]):
            support = max(support, group_vote[1])
        shortlist_support = max(support, context_support.get(value, 0))
        neighbor = evidence.neighbors[key] / max(1, sum(evidence.neighbors.values()))
        frequency = index.counts[col].get(key, 0) / max(1, len(index.keys))
        similarity = SequenceMatcher(None, old.casefold(), value.casefold(), autojunk=False).ratio()
        length_similarity = 1 - abs(len(old) - len(value)) / max(1, len(old), len(value))
        available_distance, distance = typed_distance(field, old, value)
        valid = not field.issues(value)
        admissible = value == old or valid and not any(
            hit.rule.kind == "relation" and hit.rule.id not in old_relations for hit in hits)
        vector = [float(value == old), similarity, length_similarity, float(shape(old) == shape(value)),
                  frequency, neighbor, support, reduction, min(1, strength / 4), float(valid),
                  float(not key), float(available_distance), distance, float("transform" in sources),
                  float("domain" in sources)]
        heuristic = (0.8 * similarity + 1.3 * neighbor + 1.2 * shortlist_support + 1.6 * reduction
                     + 0.3 * frequency + 0.25 * ("transform" in sources) + 0.4 * (value == old))
        result.append(Candidate(value, sources, vector, heuristic, admissible))
    original, *others = result
    others.sort(key=lambda x: (not x.admissible, direct is not None and x.value != direct[0],
                               -x.heuristic, x.value))
    return [original, *others[:settings["max_candidates"] - 1]]


def ask_pair(teacher, context, a, b):
    delta = a.heuristic - b.heuristic
    request = {"evidence": context, "A": {"value": a.value, "sources": a.sources},
               "B": {"value": b.value, "sources": b.sources}}
    return teacher.ask("repair", request, fixture={
        "winner": "unknown" if abs(delta) <= 1e-9 else "A" if delta > 0 else "B",
        "confidence": 0.9,
    })


def time_gate(original, best, policy, evidence, vote, missing_tokens=()):
    """Use corroboration and lossless extraction in place of an incompatible score margin."""
    before, after = time_minutes(original.value, extract=True), time_minutes(best.value)
    if after is None:
        return False, "time_invalid_candidate"
    if "transform" in best.sources and before == after:
        return True, "time_lossless_extraction"
    missing = (original.features[FEATURES.index("missing")] == 1.0
               or original.value.strip().casefold() in missing_tokens)
    rule_support = max((hit.support for hit in evidence.signals
                        if hit.expected and time_minutes(hit.expected) == after), default=0)
    if before is None and not missing:
        return None
    limits = policy["missing" if missing else "valid"]
    if vote and time_minutes(vote[0]) == after and vote[1] >= limits["confidence"] and vote[2] >= limits["sources"]:
        return True, "time_trusted_source_consensus"
    if rule_support >= limits["rule_support"]:
        return True, "time_supported_by_rule"
    return False, "time_keep_original_without_corroboration"


def decide(cell, pool, settings, direct=None, evidence=None, vote=None):
    original, *alternatives = pool
    if direct:
        value, name, confidence = direct
        if any(item.value == value and item.admissible for item in alternatives):
            return Repair(cell, original.value, value, value, confidence, confidence,
                          True, "direct:" + name, "direct:" + name)
    alternatives = sorted((item for item in alternatives if item.admissible),
                          key=lambda item: (-item.score, item.value))
    best = alternatives[0] if alternatives else original
    opponent = max([original.score, *(item.score for item in alternatives[1:])])
    margin = best.score - opponent
    policy = settings.get("time_gate", {}).get(cell.column)
    gated = (time_gate(original, best, policy, evidence, vote, settings.get("time_gate_missing_tokens", ()))
             if alternatives and policy else None)
    if alternatives and cell.column in settings.get("format_only_columns", ()):
        format_rule = settings.get("candidate_formats", {}).get(cell.column)
        same_value = (format_rule is not None
                      and replacement(format_rule, original.value) == replacement(format_rule, best.value))
        if same_value:
            gated = False, "keep_equivalent_format"
        else:
            is_transform = "transform" in best.sources
            gated = is_transform, "format_restoration" if is_transform else "keep_original_without_format"
    if not alternatives:
        reason = "no_admissible_candidate"
    elif settings.get("model_repair_columns") is not None and cell.column not in settings["model_repair_columns"]:
        reason = "model_repair_disabled"
    elif gated is not None:
        accepted, reason = gated
        return Repair(cell, original.value, best.value, best.value if accepted else original.value,
                      best.score, margin, accepted, reason, "|".join(best.sources))
    elif margin <= 0 or margin < settings["margin"]:
        reason = "insufficient_margin"
    else:
        reason = "accepted"
    accepted = reason == "accepted"
    return Repair(cell, original.value, best.value, best.value if accepted else original.value,
                  best.score, margin, accepted, reason, "|".join(best.sources))


def enforce_row_constraints(index, decisions):
    """Commit high-margin changes first without introducing a new relational violation."""
    by_column = {col: [rule for rule in index.relations
                       if col in (rule.target, rule.parameters["other"])]
                 for col in index.table.columns}
    committed = {}
    for decision in sorted((row for row in decisions if row.accepted),
                           key=lambda row: (-row.margin, row.cell)):
        cell = decision.cell

        def value(col):
            key = Cell(cell.row_id, col)
            return committed.get(key, index.table.at(key))

        for rule in by_column[cell.column]:
            other = rule.parameters["other"]
            before = index.relation_violation(rule, value(rule.target), value(other))
            target_value = decision.repaired_value if rule.target == cell.column else value(rule.target)
            other_value = decision.repaired_value if other == cell.column else value(other)
            if not before and index.relation_violation(rule, target_value, other_value):
                decision.accepted = False
                decision.reason = "row_constraint:" + rule.id
                decision.repaired_value = decision.dirty_value
                break
        if decision.accepted:
            committed[cell] = decision.repaired_value


def repair_scope(index, predictions, settings):
    allowed = [(pred.cell, pred.score) for pred in predictions if pred.error]
    if any(not index.fields[cell.column].focus for cell, _ in allowed):
        raise ValueError("Detection output attempts to repair a context-only column")
    if settings["expand_consensus"]:
        seen = {cell for cell, _ in allowed}
        expanded = set(index.consensus_cells(settings["expansion_confidence"]))
        allowed.extend((cell, 1.0) for cell in sorted(expanded - seen))
    return allowed


def repair(index, predictions, teacher, settings, run, output):
    archived = run.get("legacy_rule_semantics") == "archived"
    if archived and settings.get("expand_source_consensus"):
        settings = {**settings, "expand_consensus": False}
    allowed = repair_scope(index, predictions, settings)
    if not archived:
        settings = {key: value for key, value in settings.items()
                    if key not in {"model_repair_columns", "time_gate", "candidate_formats",
                                   "context_dictionaries", "format_only_columns"}}
    group_settings = settings.get("group_consensus", {}) if archived else {}
    group_votes, expand = group_consensus(index, group_settings)
    new_scope = expand - {cell for cell, _ in allowed}
    allowed.extend((cell, 1.0) for cell in sorted(new_scope))
    specs = settings.get("source_consensus", []) if archived else []
    votes = source_consensus(index, specs)
    source_scope = set()
    if archived and settings.get("expand_source_consensus"):
        seen = {cell for cell, _ in allowed}
        source_scope = {cell for cell in votes if cell not in seen and index.fields[cell.column].focus}
        allowed.extend((cell, 1.0) for cell in sorted(source_scope))
    direct = direct_repairs(index, settings.get("direct_rules", ()) if archived else (),
                            (cell for cell, _ in allowed))
    direct_scope = direct.keys() - {cell for cell, _ in allowed}
    allowed.extend((cell, 1.0) for cell in sorted(direct_scope))
    evidence = [index.evidence(cell) for cell, _ in allowed]
    dictionaries = context_candidates(index, settings.get("context_dictionaries", ()),
                                      settings.get("candidate_formats", {}))
    pools = [candidate_set(index, item, settings, votes.get(item.cell), group_votes.get(item.cell),
                           direct.get(item.cell), dictionaries.get(item.cell.column, ())) for item in evidence]
    eligible = [i for i, pool in enumerate(pools)
                if len(pool) > 1 and pool[1].admissible and allowed[i][0] not in direct_scope]
    replay_pairs = None
    archived_feedback = {}
    if teacher.mode == "replay":
        replay_pairs = {i: teacher.repair_pairs(allowed[i][0], pools[i]) for i in eligible}
        archived_feedback = {i: pairs for i in eligible
                             if (pairs := teacher.repair_pairs(allowed[i][0], pools[i], "feedback"))}
        eligible = [i for i in eligible if replay_pairs[i]]
    training, training_positions, requests, iterations = [], [], [], []
    queried = set()
    score_source = "heuristic"
    seed = run["seed"]
    rng = random.Random(seed)

    def annotate(positions, phase):
        for i in positions:
            if replay_pairs is not None:
                pairs = ([(a, b, "feedback") for a, b in archived_feedback.get(i, [])]
                         if phase == "feedback" else [])
                pairs += [(a,b,"initial") for a,b in replay_pairs[i]]
            else:
                context = index.context(evidence[i])
                original, *others = pools[i]
                candidates = [item for item in others if item.admissible][:settings["teacher_alternatives"]]
                pairs = [(original, candidate) for candidate in candidates]
                if len(candidates) > 1:
                    pairs.extend((candidates[0], candidate) for candidate in candidates[1:])
                pairs = [(a,b,"live") for a,b in pairs]
            for a, b, source_phase in pairs:
                key = (i, a.value, b.value)
                if key in queried:
                    continue
                queried.add(key)
                if rng.getrandbits(1):
                    a, b = b, a
                answer = (teacher.repair_answer(allowed[i][0], a.value, b.value, source_phase) if replay_pairs is not None
                          else ask_pair(teacher, context, a, b))
                requests.append([allowed[i][0].row_id, allowed[i][0].column, a.value, b.value,
                                 answer["winner"], answer["confidence"], phase, source_phase])
                if answer["winner"] != "unknown" and answer["confidence"] > 0:
                    label, weight = int(answer["winner"] == "A"), answer["confidence"]
                    training.append((a.features, b.features, float(label), weight))
                    training_positions.append([*requests[-1][:4], label, weight])

    selected = diverse_sample(
        [evidence[i].vector() for i in eligible],
        [(allowed[i][0].column,) for i in eligible],
        [allowed[i][1] for i in eligible], settings["teacher_budget"], seed, settings["cluster_limit"],
    )
    sampled = {eligible[i] for i in selected}
    annotate([eligible[i] for i in selected], "initial")
    student = (Student(len(FEATURES), settings["student"], seed + 1, run["device"], mask_unseen=replay_pairs is not None)
               if eligible or archived_feedback else None)
    flat = [candidate for pool in pools for candidate in pool]
    vectors = [candidate.features for candidate in flat]
    decisions = []
    for round_id in range(settings["rounds"]):
        if training:
            left, right, labels, weights = zip(*training)
            student.fit(left, labels, weights, right)
            scores = student.scores(vectors)
            score_source = "student"
        else:
            scores = [candidate.heuristic for candidate in flat]
        for candidate, score in zip(flat, scores):
            candidate.score = score
        decisions = [decide(cell, pool, settings, direct.get(cell), item, votes.get(cell))
                     for (cell, _), pool, item in zip(allowed, pools, evidence)]
        enforce_row_constraints(index, decisions)
        iterations.append({"round": round_id + 1, "pairs": len(training),
                           "student_updates": student.steps if student else 0,
                           "accepted": sum(row.accepted for row in decisions)})
        if round_id + 1 == settings["rounds"] or settings["feedback_budget"] == 0:
            break
        remaining = set(eligible) - sampled
        remaining.update(i for i, pairs in archived_feedback.items()
                         if any((i,a.value,b.value) not in queried for a,b in pairs))
        remaining = list(remaining)
        remaining.sort(key=lambda i: (abs(decisions[i].margin - settings["margin"]), allowed[i][0]))
        batch = remaining[:settings["feedback_budget"]]
        if not batch:
            break
        sampled.update(batch)
        annotate(batch, "feedback")
    if student and student.steps:
        student.save(output / "ranker.pt", FEATURES)
    detected = {p.cell for p in predictions if p.error}
    write_csv(output / "repair_scope.csv", ["row_id", "column", "source"],
              ([cell.row_id, cell.column, "detection" if cell in detected else
                "group_consensus" if cell in new_scope else "source_consensus" if cell in source_scope
                else "direct_rule" if cell in direct_scope else "consensus"]
               for cell, _ in allowed))
    write_csv(output / "repair_candidates.csv", ["row_id", "column", "value", "sources", "score", "admissible", *FEATURES],
              ([cell.row_id, cell.column, candidate.value, "|".join(candidate.sources), candidate.score,
                int(candidate.admissible), *candidate.features]
               for (cell, _), pool in zip(allowed, pools) for candidate in pool))
    write_csv(output / "repair_pairwise.csv", ["row_id", "column", "a", "b", "winner", "confidence", "phase",
                                              "supervision_phase"], requests)
    # Round r fits the first repair_training.json.rounds[r-1].pairs rows in input order.
    write_csv(output / "repair_training_pairs.csv",
              ["row_id", "column", "a", "b", "label", "weight"], training_positions)
    write_csv(output / "repairs.csv",
              ["row_id", "column", "dirty_value", "proposed_value", "repaired_value", "score", "margin",
               "accepted", "reason", "source"],
              ([r.cell.row_id, r.cell.column, r.dirty_value, r.proposed_value, r.repaired_value,
                r.score, r.margin, int(r.accepted), r.reason, r.source] for r in decisions))
    info = {"scorer": score_source, "rounds": iterations, "features": FEATURES}
    if direct:
        info["direct_rules"] = {"proposed": len(direct), "outside_scope": len(direct_scope)}
    if specs:
        info["source_consensus"] = {"scope": sum(item.cell in votes for item in evidence),
                                    "added_scope": len(source_scope),
                                    "shortlisted_cells": sum(any("source_consensus" in c.sources for c in pool)
                                                               for pool in pools)}
    if group_settings:
        info["group_consensus"] = {"proposed_cells": len(group_votes), "added_scope": len(new_scope),
                                   "shortlisted_cells": sum(any("group_consensus" in c.sources for c in pool)
                                                            for pool in pools)}
    if student is not None and student.active is not None:
        info["masked_features"] = [name for name, used in zip(FEATURES, student.active.tolist()) if not used]
    write_json(output / "repair_training.json", info)
    return decisions
