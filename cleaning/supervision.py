"""Use verified historical labels as supervision, without a model client."""

import csv
import io
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from .data import Cell, Table, digest
from .teacher import validate


def archive_text(value):
    return " ".join(value.split()).casefold()


def feedback_paths(artifacts, recorded=False):
    """Choose published feedback artifacts and ignore the overlapping last-round-only file."""
    detection, repair = [], []
    for name in sorted(artifacts):
        path = Path(name)
        if name.startswith("detection/") and (path.name == "hard_cases_relabeled.jsonl"
                or re.fullmatch(r"iter_\d+_relabeled_hard_cases\.jsonl", path.name)
                or recorded and re.fullmatch(r"iter_\d+_llm_relabels\.jsonl", path.name)):
            detection.append(path)
        elif name.startswith("correction/") and path.name == "pairwise_responses_accumulated.jsonl":
            repair.append(path)
    if len(repair) > 1:
        raise ValueError("Ambiguous accumulated repair feedback files")
    return detection, repair


def schedule_paths(artifacts):
    return [Path(name) for name in sorted(artifacts) if name.startswith("detection/")
            and re.fullmatch(r"random_feedback_selection_iter_\d+\.json", Path(name).name)]


def feedback_round(path):
    value = path.parent.name if path.name == "hard_cases_relabeled.jsonl" else path.name
    match = re.match(r"iter_(\d+)(?:_|$)", value)
    if not match:
        raise ValueError(f"Missing feedback iteration: {path}")
    return int(match[1]) - 1


class ArchivedTeacher:
    mode = "replay"

    def __init__(self, archive: Path, dataset, source: Table, dirty: Table, row_map=None, *,
                 fields=None, include_feedback=True, recorded_detection=False):
        archive = archive.resolve()
        manifest_file = archive / "run_manifest.json"
        manifest_bytes = manifest_file.read_bytes()
        manifest = json.loads(manifest_bytes)
        identity = manifest["identity"]
        if (manifest["status"] != "complete" or identity["dataset"] != dataset
                or identity["inputs"]["dirty"]["sha256"] != source.sha256):
            raise ValueError("Archive does not match the dataset and dirty source")
        mapping = source_to_run(source, dirty, row_map)
        detection_sources = list((archive / "detection").glob("candidate_sampled_labeled_*.csv"))
        if len(detection_sources) != 1:
            raise ValueError("Expected one archived initial detection label CSV")
        labels = verified_bytes(archive, detection_sources[0], manifest["artifacts"])
        pair_file = archive / "correction" / "repair_pairwise_responses_v2.jsonl"
        pairs = verified_bytes(archive, pair_file, manifest["artifacts"])
        self.provenance = {
            "run": str(archive), "manifest_sha256": digest(manifest_bytes), "dirty_source_sha256": source.sha256,
            "detection_labels_sha256": digest(labels), "pairwise_sha256": digest(pairs),
            "model_requested": identity["llm_model_override"],
            "row_map_sha256": digest(row_map.read_bytes()) if row_map else None,
        }
        self.requests = self.hits = self.misses = 0
        self.served_models = set()
        self.counts = Counter()
        self.fields = fields or {}
        self.labels = {}
        self.feedback = {}
        self.recorded_detection = recorded_detection
        self.feedback_rounds = defaultdict(dict)
        self.preferences = defaultdict(dict)
        self.repair_feedback = defaultdict(dict)
        self.bound_preferences = {}
        for row in csv.DictReader(io.StringIO(labels.decode("utf-8-sig"), newline="")):
            cell = self.cell(row, "value", mapping, dirty, "detection")
            if cell is None:
                continue
            name = str(row["is_error"]).strip().lower()
            if name not in {"true", "false", "1", "0", "1.0", "0.0"}:
                self.counts["detection_unusable"] += 1
                continue
            label = "error" if name in {"true", "1", "1.0"} else "correct"
            try:
                confidence = float(row["confidence"])
                outcome = validate("detect", {"label": label, "confidence": confidence})
            except ValueError:
                self.counts["detection_unusable"] += 1
                continue
            self.record(self.labels, cell, outcome, "detection")
        self.load_repairs(pairs, self.preferences, mapping, dirty, "repair")
        if include_feedback:
            detection_feedback, repair_feedback = feedback_paths(manifest["artifacts"], recorded_detection)
            schedule = self.load_schedule(archive, identity, manifest["artifacts"]) if recorded_detection else {}
            self.load_feedback(detection_feedback, archive, manifest["artifacts"], mapping, dirty, schedule)
            self.provenance["repair_feedback_files"] = {p.as_posix(): manifest["artifacts"][p.as_posix()]
                                                         for p in repair_feedback}
            for path in repair_feedback:
                self.load_repairs(verified_bytes(archive, archive/path, manifest["artifacts"]),
                                  self.repair_feedback, mapping, dirty, "repair_feedback")
        self.counts["detection_available"] = sum(answer is not None for answer in self.labels.values())
        self.counts["repair_available_pairs"] = sum(winner is not None for values in self.preferences.values()
                                                    for winner in values.values())
        self.counts["repair_feedback_available_pairs"] = sum(winner is not None for values in self.repair_feedback.values()
                                                             for winner in values.values())

    def load_repairs(self, pairs, target, mapping, dirty, task):
        for line in pairs.decode("utf-8-sig").split("\n"):
            if not line.strip():
                continue
            row = json.loads(line)
            if task == "repair_feedback" and not re.fullmatch(r"iterative_hard_case_llm(?:_\w+)?", str(row.get("source", ""))):
                self.counts["repair_feedback_other_source"] += 1
                continue
            cell = self.cell(row, "dirty_value", mapping, dirty, task)
            if cell is None:
                continue
            a, b = row["candidate_a_value"], row["candidate_b_value"]
            winner = str(row["winner"]).strip().upper()
            if not isinstance(a, str) or not isinstance(b, str) or a == b or winner not in {"A", "B"}:
                self.counts[task + "_unusable"] += 1
                continue
            if a > b:
                a, b = b, a
                winner = "B" if winner == "A" else "A"
            self.record(target[cell], (a, b), winner, task)

    def load_schedule(self, archive, identity, artifacts):
        schedule = {}
        files = schedule_paths(artifacts)
        if not files or identity.get("feedback_selection") != "random":
            raise ValueError("No verified archived random detection schedule")
        self.provenance["detection_schedule"] = {path.as_posix(): artifacts[path.as_posix()] for path in files}
        for path in files:
            data = json.loads(verified_bytes(archive, archive / path, artifacts))
            iteration = data["iteration"]
            keys = [tuple(key) for key in data["selected_keys"]]
            if (data["dataset"] != identity["dataset"] or data["stage"] != "detection"
                    or data["base_seed"] != identity["seed"] or data["strategy"] != "random"
                    or data["uses_clean_table"] is not False or data["initial_training_changed"] is not False
                    or iteration in schedule or len(keys) != len(set(keys)) or len(keys) != data["selected_cells"]):
                raise ValueError(f"Inconsistent detection feedback schedule: {path}")
            schedule[iteration] = {key: i for i, key in enumerate(keys)}
        return schedule

    def load_feedback(self, files, archive, artifacts, mapping, dirty, schedule):
        self.provenance["feedback_files"] = {name.as_posix(): artifacts[name.as_posix()] for name in files}
        for name in files:
            data = verified_bytes(archive, archive/name, artifacts)
            iteration = feedback_round(name) if self.recorded_detection else None
            for line in data.decode("utf-8-sig").split("\n"):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("label_source") != "llm_relabel":
                    self.counts["feedback_other_source"] += 1
                    continue
                cell = self.cell(row, "value", mapping, dirty, "feedback")
                if cell is None:
                    continue
                try:
                    answer = validate("detect", {"label": row["label"], "confidence": row["confidence"]})
                except ValueError:
                    self.counts["feedback_unusable"] += 1
                    continue
                binary = {"error": 1, "correct": 0}.get(answer["label"])
                if binary is None or row.get("label_binary") != binary:
                    self.counts["feedback_unusable"] += 1
                    continue
                if iteration is not None:
                    position = schedule.get(iteration, {}).get((int(row["row_id"]), row["column"]))
                    if position is None:
                        self.counts["feedback_not_in_recorded_schedule"] += 1
                        continue
                    self.record(self.feedback_rounds[iteration], cell, (position, answer), "recorded_feedback")
                prior = self.feedback.get(cell)
                if prior and prior["label"] == answer["label"]:
                    self.counts["feedback_duplicate"] += 1
                else:
                    self.record(self.feedback, cell, answer, "feedback")
        self.counts["feedback_available"] = sum(answer is not None for answer in self.feedback.values())

    def cell(self, row, field, mapping, dirty, task):
        self.counts[task + "_rows"] += 1
        if int(row["row_id"]) not in mapping:
            self.counts[task + "_outside_scope"] += 1
            return None
        cell = Cell(mapping[int(row["row_id"])], row["column"])
        observed = row[field]
        if cell.column not in dirty.index:
            self.counts[task + "_dirty_mismatch"] += 1
            return None
        raw = dirty.at(cell)
        schema = self.fields.get(cell.column)
        missing = raw == "" or (schema is not None and not schema.key(raw))
        if not isinstance(observed, str):
            # Legacy detection feedback serializes pandas missing values as JSON NaN.
            is_null = observed is None or isinstance(observed, float) and math.isnan(observed)
            if task == "feedback" and is_null and missing:
                self.counts["feedback_dirty_missing_alias"] += 1
                return cell
            self.counts[task + "_dirty_mismatch"] += 1
            return None
        if observed != raw:
            if (archive_text(raw) != archive_text(observed)
                    and not (archive_text(observed) in {"", "empty"} and missing)):
                self.counts[task + "_dirty_mismatch"] += 1
                return None
            self.counts[task + "_dirty_normalized"] += 1
        return cell

    def record(self, records, key, value, task):
        if key not in records:
            records[key] = value
        elif records[key] is None:
            return
        elif records[key] == value:
            self.counts[task + "_duplicate"] += 1
        else:
            records[key] = None
            self.counts[task + "_conflict"] += 1

    def detection_indices(self, evidence, *, archive_order=False):
        if archive_order:
            positions = {item.cell: i for i, item in enumerate(evidence)}
            indices = [positions[cell] for cell, label in self.labels.items()
                       if label is not None and cell in positions]
        else:
            indices = [i for i, item in enumerate(evidence) if self.labels.get(item.cell) is not None]
        self.counts["detection_candidate_matches"] = len(indices)
        return indices

    def feedback_indices(self, evidence, initial):
        if not self.feedback:
            return initial
        indices = [i for i, item in enumerate(evidence) if self.feedback.get(item.cell) is not None]
        self.counts["feedback_candidate_matches"] = len(indices)
        return sorted(set(initial) | set(indices))

    def detection_feedback_batches(self, evidence):
        positions = {item.cell: i for i, item in enumerate(evidence)}
        batches = []
        for round_no, records in sorted(self.feedback_rounds.items()):
            matched = [(record[0], positions[cell]) for cell, record in records.items()
                       if cell in positions and record is not None]
            indices = [index for _, index in sorted(matched)]
            self.counts["recorded_feedback_round_" + str(round_no)] = len(indices)
            if indices:
                batches.append((round_no, indices))
        return batches

    def detection_answer(self, cell, phase="initial", original_round=None):
        self.counts["detection_used"] += 1
        if original_round is not None:
            self.counts["feedback_used"] += 1
            return self.feedback_rounds[original_round][cell][1]
        if phase == "feedback" and self.feedback.get(cell) is not None:
            self.counts["feedback_used"] += 1
            return self.feedback[cell]
        return self.labels[cell]

    def repair_pairs(self, cell, pool, phase="initial"):
        allowed = {candidate.value: candidate for candidate in pool if candidate.admissible}
        keys = (self.repair_feedback if phase == "feedback" else self.preferences).get(cell, {})
        prefix = "repair_feedback" if phase == "feedback" else "repair"
        aliases = defaultdict(list)
        for value in allowed:
            aliases[archive_text(value)].append(value)

        def resolve(value):
            if value in allowed:
                return value
            normalized = archive_text(value)
            options = aliases[normalized]
            # Blank representations are never collapsed for candidates. Display variants
            # are safe only if the current pool has an unambiguous textual representative.
            return options[0] if normalized and len(options) == 1 else None

        bound = {}
        for (a, b), winner in sorted(keys.items()):
            if winner is None:
                continue
            x, y = resolve(a), resolve(b)
            if x is None or y is None or x == y:
                continue
            if (x, y) != (a, b):
                self.counts[prefix + "_pair_normalized"] += 1
            if x > y:
                x, y = y, x
                winner = "B" if winner == "A" else "A"
            self.record(bound, (x, y), winner, prefix + "_bound")
        if phase == "feedback":
            initial = self.bound_preferences.get((cell,"initial"), {})
            for key in bound.keys() & initial.keys():
                if initial[key] != bound[key]:
                    self.counts["repair_feedback_conflicting_initial"] += 1
                    bound[key] = None
        self.bound_preferences[(cell,phase)] = bound
        pairs = [(allowed[a], allowed[b]) for (a, b), winner in sorted(bound.items()) if winner is not None]
        self.counts[prefix + "_candidate_pair_matches"] += len(pairs)
        return pairs

    def repair_answer(self, cell, a, b, phase="initial"):
        winner = self.bound_preferences[(cell,phase)][tuple(sorted((a, b)))]
        if a > b:
            winner = "B" if winner == "A" else "A"
        self.counts["repair_used"] += 1
        if phase == "feedback":
            self.counts["repair_feedback_used"] += 1
        # These historical files store categorical preferences without confidences.
        return {"winner": winner, "confidence": 1.0}


def verified_bytes(archive, path, artifacts):
    data = path.read_bytes()
    if artifacts.get(path.relative_to(archive).as_posix()) != digest(data):
        raise ValueError(f"Archived supervision hash mismatch: {path.name}")
    return data


def source_to_run(source, dirty, row_map):
    if row_map is None:
        if source.sha256 != dirty.sha256:
            raise ValueError("A subset requires an explicit, verified source row map")
        return {i: i for i in range(len(dirty.rows))}
    source_rows, sample_rows = set(), set()
    mapping = {}
    if source.columns != dirty.columns:
        raise ValueError("Replay source and sample columns differ")
    with row_map.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["sample_row_id", "source_row_id"]:
            raise ValueError("Invalid replay row map columns")
        for row in reader:
            a, b = int(row["source_row_id"]), int(row["sample_row_id"])
            if (a in source_rows or b in sample_rows or not 0 <= a < len(source.rows)
                    or not 0 <= b < len(dirty.rows) or source.rows[a] != dirty.rows[b]):
                raise ValueError("Invalid or mismatched replay row mapping")
            source_rows.add(a)
            sample_rows.add(b)
            mapping[a] = b
    if len(sample_rows) != len(dirty.rows):
        raise ValueError("Replay row map must cover every input row")
    return mapping
