from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import random

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .data import Cell, write_csv, write_json
from .evidence import FEATURE_NAMES, sigmoid
from .models import Student, class_weights, verifier_fit
from .sampling import diverse_sample


FEATURES = list(FEATURE_NAMES)


@dataclass
class Prediction:
    cell: Cell
    dirty_value: str
    score: float
    detector_score: float
    error: bool


def is_calibration(cell, seed, fraction):
    key = f"{seed}:{cell.row_id}:{cell.column}".encode()
    value = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / 2**64
    return value < fraction


def teacher_label(teacher, index, evidence, phase, original_round=None):
    if teacher.mode == "replay":
        answer = teacher.detection_answer(evidence.cell, phase, original_round)
    else:
        payload = {"cell": asdict(evidence.cell), "evidence": index.context(evidence), "phase": phase}
        answer = teacher.ask("detect", payload, fixture={
            "label": "error" if evidence.invalid or evidence.probability >= 0.65 else "correct",
            "confidence": 0.9,
        })
    if answer["label"] == "unknown":
        return None
    return int(answer["label"] == "error"), answer["confidence"]


def distance_margin_propagate(vectors, columns, labels, settings, *, blocked=(), standardized=False):
    """Teacher-only, inverse-distance KNN with an explicit confidence margin."""
    if len(labels) < settings["minimum_teachers"]:
        return {}
    values = np.asarray(vectors, dtype=float)
    values = values if standardized else StandardScaler().fit_transform(values)
    groups = defaultdict(list)
    for i, col in enumerate(columns):
        groups[col].append(i)
    blocked = set(blocked)
    result = {}
    k = settings["neighbors"]
    for members in groups.values():
        teachers = [i for i in members if i in labels]
        targets = [i for i in members if i not in labels and i not in blocked]
        if len(teachers) < max(3, min(k, settings["minimum_teachers"] // 4)) or not targets:
            continue
        search = NearestNeighbors(n_neighbors=min(k, len(teachers)), metric="euclidean").fit(values[teachers])
        for target in targets:
            distances, positions = search.kneighbors(values[target:target+1])
            distances, positions = distances[0], positions[0]
            if float(distances.mean()) > settings["maximum_mean_distance"]:
                continue
            selected = np.array([labels[teachers[i]][0] for i in positions])
            weights = 1 / (distances + settings["distance_epsilon"])
            positive = float(weights[selected == 1].sum())
            negative = float(weights[selected == 0].sum())
            total = positive + negative
            if total <= 0:
                continue
            label = int(positive >= negative)
            confidence = (positive if label else negative) / total
            margin = abs(positive-negative) / total
            if (confidence >= settings["threshold"] and margin >= settings["margin"]
                    and confidence >= settings["minimum_confidence"]):
                result[target] = (label, round(settings["sample_weight"] * confidence, 6))
    return result


def propagate(vectors, evidence, labels, minimum, settings=None):
    """Confidence-weighted, same-column nearest teacher votes."""
    if settings:
        if settings.get("algorithm") != "distance_margin":
            raise ValueError(f"Unknown propagation algorithm: {settings.get('algorithm')}")
        return distance_margin_propagate(vectors, [item.cell.column for item in evidence], labels, settings)
    additions = {}
    groups = defaultdict(list)
    for i, item in enumerate(evidence):
        groups[item.cell.column].append(i)
    for positions in groups.values():
        trained = [i for i in positions if i in labels]
        unlabelled = [i for i in positions if i not in labels]
        if len(trained) < 3 or not unlabelled:
            continue
        scaler = StandardScaler().fit(vectors[trained])
        search = NearestNeighbors(n_neighbors=min(5, len(trained))).fit(scaler.transform(vectors[trained]))
        distances, neighbors = search.kneighbors(scaler.transform(vectors[unlabelled]))
        for row, i in enumerate(unlabelled):
            votes = Counter()
            for distance, position in zip(distances[row], neighbors[row]):
                label, confidence = labels[trained[position]]
                votes[label] += confidence / (1 + distance)
            vote_total = sum(votes.values())
            if vote_total:
                label = max(votes, key=votes.get)
                confidence = votes[label] / vote_total
                if confidence >= minimum and float(distances[row].mean()) <= 2.5:
                    additions[i] = (label, confidence * 0.5)
    return additions


def pseudo_negatives(index, candidates, count, seed, calibration_fraction):
    """Sample low-conflict, repeated dirty-table values as low-weight negatives."""
    if count <= 0:
        return []
    rng = random.Random(seed)
    positions = [(index.table.index[col], col) for col, field in index.fields.items() if field.focus]
    rng.shuffle(positions)
    result = []
    if not positions:
        return result
    for rid in rng.sample(range(len(index.table.rows)), min(4096, len(index.table.rows))):
        row = index.table.rows[rid]
        for j, col in positions:
            key = index.keys[rid][j]
            cell = Cell(rid, col)
            if (not key or index.counts[col][key] < 3 or cell in candidates
                    or is_calibration(cell, seed, calibration_fraction) or index.check(cell, row[j])):
                continue
            evidence = index.evidence(cell, [])
            if evidence.neighbors and evidence.neighbors[key] >= sum(evidence.neighbors.values()) / 2:
                result.append((cell, evidence.vector()))
                if len(result) == count:
                    return result
    return result


def evidence_veto(item, settings):
    guards = settings.get("reject_primary_signals", {})
    primary = item.signals[0].rule.id
    if primary in guards and not any(hit.rule.kind in guards[primary] for hit in item.signals[1:]):
        return primary
    return ""


def detect(index, teacher, settings, run, output):
    feature_names = list(index.feature_names)
    feedback_student = settings["feedback_student"]
    if feedback_student not in {"update", "freeze"}:
        raise ValueError(f"Invalid detector feedback policy: {feedback_student}")
    evidence = [index.evidence(cell, hits) for cell, hits in index.candidate_cells()]
    if index.archive_scope and settings.get("context", {}).get("candidate_order") == "conflict":
        evidence.sort(key=index.candidate_sort_key)
    count = len(evidence)
    vectors = np.asarray([item.vector() for item in evidence], dtype=np.float32).reshape(count, len(feature_names))
    risks = [item.probability for item in evidence]
    groups = [(item.cell.column, item.signals[0].rule.kind) for item in evidence]
    seed = run["seed"]
    recorded = teacher.mode == "replay" and teacher.recorded_detection
    recorded_batches = teacher.detection_feedback_batches(evidence) if recorded else []
    if teacher.mode == "replay":
        initial_replay = settings["initial_replay"]
        if initial_replay == "archive":
            # The archived teacher CSV is already the result of historical sampling.
            available = samples = teacher.detection_indices(evidence, archive_order=True)
        elif initial_replay == "resample":
            available = teacher.detection_indices(evidence)
            samples = diverse_sample(vectors[available], [groups[i] for i in available], [risks[i] for i in available],
                                     settings["teacher_budget"], seed, settings["cluster_limit"])
            samples = [available[i] for i in samples]
        else:
            raise ValueError(f"Invalid detector initial replay policy: {initial_replay}")
        feedback_available = teacher.feedback_indices(evidence, available)
    else:
        available = range(count)
        feedback_available = available
        samples = diverse_sample(vectors, groups, risks, settings["teacher_budget"], seed, settings["cluster_limit"])
    labels, queried = {}, set()
    label_history, iterations = [], []
    propagation_history, training_history = [], []
    contract_conflicts = []

    def annotate(positions, phase, original_round=None):
        for i in positions:
            queried.add(i)
            label = teacher_label(teacher, index, evidence[i], phase, original_round)
            match = index.contracts.matches.get(evidence[i].cell)
            conflict = match is not None and label is not None and label[0] == 0
            if conflict:
                contract_conflicts.append([evidence[i].cell.row_id, evidence[i].cell.column, phase,
                                           match.operation_id, label[0], label[1], match.teacher_conflicts])
                if match.teacher_conflicts == "quarantine":
                    labels.pop(i, None)
            if label is not None:
                if not conflict or match.teacher_conflicts != "quarantine":
                    labels[i] = label
            label_history.append([evidence[i].cell.row_id, evidence[i].cell.column, phase,
                                  None if label is None else label[0], None if label is None else label[1]])

    annotate(samples, "initial")
    student_settings = settings["student"]
    if index.encoding.names:
        student_settings = {**student_settings, "unscaled_from": len(index.numeric_feature_names), "mask_unseen": True}
        write_json(output / "detection_encoding.json", index.encoding.snapshot())
    student = Student(len(feature_names), student_settings, seed, run["device"]) if count else None
    calibrated = {i for i, item in enumerate(evidence)
                  if is_calibration(item.cell, seed, settings["calibration_fraction"])}
    final, detector_scores, keep, mode = risks, risks, [1.0] * count, "heuristic"
    verifier = None
    negatives = None
    candidates = {item.cell for item in evidence}
    rounds = 1 + len(recorded_batches) if recorded else settings["rounds"]
    for round_id in range(rounds if count else 0):
        train = {i: label for i, label in labels.items() if i not in calibrated and label[1] > 0}
        # Exclude calibration cells from both the voting pool and the training targets.
        propagated = {i: label for i, label in propagate(vectors, evidence, train, settings["propagation_threshold"],
                                                       settings.get("propagation")).items()
                      if i not in calibrated and i not in queried
                      and not (label[0] == 0 and evidence[i].cell in index.contracts.matches
                               and index.contracts.matches[evidence[i].cell].teacher_conflicts == "quarantine")}
        train.update(propagated)
        propagation_history.extend([round_id + 1, evidence[i].cell.row_id, evidence[i].cell.column, label, weight]
                                   for i, (label, weight) in propagated.items())
        training_vectors = vectors[list(train)]
        training_cells = [evidence[i].cell for i in train]
        sources = ["propagated" if i in propagated else "teacher" for i in train]
        observed = [label for label, _ in train.values()]
        weights = [weight for _, weight in train.values()]
        pseudo_count = 0
        if 1 in observed:
            if negatives is None:
                negatives = pseudo_negatives(index, candidates, settings["pseudo_negative_budget"],
                                             seed, settings["calibration_fraction"])
            pseudo_count = min(len(negatives), observed.count(1))
            if pseudo_count:
                cells, negative_vectors = zip(*negatives[:pseudo_count])
                training_vectors = np.concatenate([training_vectors, negative_vectors], axis=0)
                training_cells.extend(cells)
                sources.extend(["outside_candidate"] * pseudo_count)
                observed += [0] * pseudo_count
                weights += [0.25] * pseudo_count
        can_fit_student = feedback_student == "update" or not student.steps
        if can_fit_student and len(set(observed)) == 2:
            weights = class_weights(observed, weights)
            training_history.extend([round_id + 1, cell.row_id, cell.column, label, source, weight, *vector.tolist()]
                                    for cell, label, source, weight, vector in
                                    zip(training_cells, observed, sources, weights, training_vectors))
            student.fit(training_vectors, observed, weights)
            detector_scores = [sigmoid(score) for score in student.scores(vectors)]
            mode = "student"
        elif student.steps:
            detector_scores = [sigmoid(score) for score in student.scores(vectors)]
            mode = "student"
        verifier = verifier_fit(sorted(set(labels) & calibrated), labels, vectors, detector_scores,
                                settings["threshold"], seed, settings["verifier"]) if mode == "student" else None
        keep = [1.0] * count
        if verifier:
            samples = np.column_stack([vectors, detector_scores])
            keep = verifier.predict_proba(samples)[:, list(verifier.classes_).index(1)].tolist()
        final = [score if probability >= settings["verifier_threshold"] else 0.0
                 for score, probability in zip(detector_scores, keep)]
        iterations.append({"round": round_id + 1, "training_labels": len(train),
                           "pseudo_negatives": pseudo_count,
                           "teacher_labels": len(labels), "calibration_labels": len(set(labels) & calibrated),
                           "student_updates": student.steps, "verifier": verifier is not None})
        if feedback_student == "freeze":
            iterations[-1]["student_frozen"] = not can_fit_student
        if round_id + 1 == rounds:
            break
        if recorded:
            original_round, indices = recorded_batches[round_id]
            annotate(indices, "feedback", original_round)
            iterations[-1]["archive_feedback_round"] = original_round
            continue
        if settings["feedback_budget"] == 0:
            break
        remaining = [i for i in feedback_available if i not in queried]
        remaining.sort(key=lambda i: (min(abs(detector_scores[i] - settings["threshold"]),
                                         abs(keep[i] - settings["verifier_threshold"])), -risks[i], i))
        if not remaining:
            break
        annotate(remaining[:settings["feedback_budget"]], "feedback")
    if student and student.steps:
        student.save(output / "detector.pt", feature_names)
    if count and verifier is not None:
        import joblib
        joblib.dump(verifier, output / "verifier.joblib")
    threshold = settings["threshold"]
    if run.get("legacy_rule_semantics") == "archived" and settings.get("reject_primary_signals"):
        decisions = []
        for i, item in enumerate(evidence):
            veto = evidence_veto(item, settings)
            if final[i] >= threshold and veto:
                final[i] = 0.0
                decisions.append((item.cell.row_id, item.cell.column, veto, detector_scores[i]))
        write_csv(output / "detection_policy_vetoes.csv", ["row_id", "column", "primary_rule", "raw_score"],
                  decisions)
    if settings.get("valid_value_guards"):
        guarded = []
        for i, item in enumerate(evidence):
            guard = index.valid_value_guards.veto(item.cell, index.table.at(item.cell), item.signals)
            if guard and final[i] >= threshold and item.cell not in index.contracts.matches:
                guarded.append([item.cell.row_id, item.cell.column, guard, final[i], detector_scores[i]])
                final[i] = 0.0
        write_csv(output / "detection_valid_value_guards.csv",
                  ["row_id", "column", "guard_id", "before_score", "student_score"], guarded)
        run["valid_value_guards"] = {"suppressed_predictions": len(guarded)}
    predictions = [
        Prediction(item.cell, index.table.at(item.cell), final[i], detector_scores[i], final[i] >= threshold)
        for i, item in enumerate(evidence)
    ]
    if settings.get("canonical_contracts"):
        contract_decisions = []
        for prediction in predictions:
            match = index.contracts.matches.get(prediction.cell)
            if match is not None:
                before_score, before_error = prediction.score, prediction.error
                prediction.score = max(prediction.score, match.confidence)
                prediction.error = True
                contract_decisions.append([prediction.cell.row_id, prediction.cell.column, match.operation_id,
                                           match.expected, prediction.detector_score, before_score,
                                           prediction.score, int(not before_error)])
        write_csv(output / "detection_contract_decisions.csv",
                  ["row_id", "column", "operation_id", "expected", "student_score", "before_score",
                   "score", "overridden_negative"], contract_decisions)
        write_csv(output / "detection_contract_conflicts.csv",
                  ["row_id", "column", "phase", "operation_id", "label", "confidence", "action"],
                  contract_conflicts)
        run["canonical_contracts"] = {"matched_cells": len(contract_decisions),
                                      "overridden_negatives": sum(row[-1] for row in contract_decisions),
                                      "teacher_conflict_events": len(contract_conflicts),
                                      "quarantined_events": sum(row[-1] == "quarantine" for row in contract_conflicts)}
    write_csv(output / "detection_labels.csv", ["row_id", "column", "phase", "label", "confidence"], label_history)
    write_csv(output / "detection_propagation.csv", ["round", "row_id", "column", "label", "weight"], propagation_history)
    write_csv(output / "detection_training_rows.csv",
              ["round", "row_id", "column", "label", "source", "weight", *feature_names], training_history)
    write_csv(output / "detection_evidence.csv",
              ["row_id", "column", *feature_names, "rules", "verifier_keep_probability"],
              ([item.cell.row_id, item.cell.column, *item.vector(),
                "|".join(hit.rule.id for hit in item.signals), keep[i]] for i, item in enumerate(evidence)))
    if index.typed_context is not None:
        keys = ["semantic_type", "value_frequency", "value_frequency_rank", "non_null_count",
                "neighbor_majority_value", "neighbor_majority_ratio", "violation_count", "conflict_score",
                "prior_error_probability", "posterior_error_probability",
                "rule_signal", "rarity_signal", "pattern_signal", "neighbor_signal"]
        write_csv(output / "detection_context.csv", ["row_id", "column", *keys],
                  ([item.cell.row_id, item.cell.column, *(item.context_summary.get(key) for key in keys)]
                   for item in evidence))
    write_json(output / "detection_training.json", {"scorer": mode, "rounds": iterations, "features": feature_names})
    return predictions
