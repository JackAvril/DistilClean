from __future__ import annotations

import argparse
import importlib.metadata
import platform
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

from . import __version__
from .data import Table, digest, fields_for, read_json, write_csv, write_json
from .detection import detect
from .evidence import EvidenceIndex
from .repair import repair
from .results import read_predictions, write_predictions, write_repaired
from .rules import load_pool
from .configuration import config_for
from .supervision import ArchivedTeacher
from .contracts import contract_operations
from .teacher import Teacher, encoded


ROOT = Path(__file__).resolve().parents[1]




def evidence_fingerprint(config, rules_sha256, dataset, import_policy=None):
    detection = config["detection"]
    identity = {
        "dataset": dataset, "rules_sha256": rules_sha256,
        "import_policy": import_policy or {},
        "settings": {key: detection[key] for key in
                     ("neighbors", "neighbor_pool", "rare_max_count", "rare_pattern_ratio")},
        "context": detection.get("context", {}),
        "canonical_contracts": detection.get("canonical_contracts", []),
        "valid_value_guards": detection.get("valid_value_guards", []),
        "contract_operations": contract_operations(config),
        "code": {name: digest((Path(__file__).parent / name).read_bytes())
                     for name in ("data.py", "rules.py", "evidence.py", "context.py", "missing.py", "encoding.py",
                                  "contracts.py", "repair_rules.py")},
    }
    return digest(encoded(identity).encode("utf-8"))


def parse_args():
    parser = argparse.ArgumentParser(description="Splittree detection and repair")
    parser.add_argument("mode", choices=("detection", "correction", "all"))
    parser.add_argument("--dataset", required=True, choices=("hospital", "beers", "flights", "adult", "rayyan", "soccer"))
    parser.add_argument("--output", type=Path, required=True, help="New run directory")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True, help="Complete external experiment configuration")
    parser.add_argument("--dirty", type=Path)
    parser.add_argument("--detections", type=Path, help="Detection run directory, required for correction")
    parser.add_argument("--verifier", action=argparse.BooleanOptionalAction, default=None,
                        help="Override detection.verifier_enabled for detection or all")
    parser.add_argument("--accept-legacy-detections", action="store_true",
                        help="Allow a schema 1 detection artifact for a controlled correction comparison")
    parser.add_argument("--rule-pool", type=Path, required=True, help="External rule-pool JSON")
    parser.add_argument("--legacy-rule-semantics", choices=("conservative", "archived"), required=True,
                        help="Opt in to source-version candidate generation when importing a legacy pool")
    parser.add_argument("--teacher", choices=("cache", "fixture", "live", "replay"), required=True)
    parser.add_argument("--archive-run", type=Path, help="Completed historical run with verified teacher artifacts")
    parser.add_argument("--archive-feedback", action=argparse.BooleanOptionalAction, default=None,
                        help="Use verified archive labels in feedback rounds (default in replay)")
    parser.add_argument("--recorded-detection-feedback", action="store_true",
                        help="Replay verified detection labels in their archived iteration and order")
    parser.add_argument("--source-row-map", type=Path, help="Verified projection from original dirty rows to this run")
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--endpoint")
    parser.add_argument("--max-requests", type=int, help="Explicit live API request cap including retries")
    args = parser.parse_args()
    if args.endpoint:
        endpoint = urlsplit(args.endpoint)
        if endpoint.scheme not in {"http", "https"} or not endpoint.hostname or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment:
            parser.error("--endpoint must be an HTTP(S) base URL without credentials, query or fragment")
    if args.mode == "correction" and args.detections is None:
        parser.error("correction requires --detections")
    if args.mode != "correction" and args.detections is not None:
        parser.error("--detections is only used by correction")
    if args.mode == "correction" and args.verifier is not None:
        parser.error("--verifier/--no-verifier only applies to detection or all")
    if args.mode != "correction" and args.accept_legacy_detections:
        parser.error("--accept-legacy-detections is only used by correction")
    if args.legacy_rule_semantics == "archived" and args.rule_pool is None:
        parser.error("archived rule semantics requires --rule-pool")
    if args.teacher == "replay":
        if not args.archive_run or any((args.model, args.endpoint, args.cache, args.max_requests)):
            parser.error("replay requires --archive-run and disallows live or request-cache options")
        if args.recorded_detection_feedback and (args.archive_feedback is False or args.mode == "correction"):
            parser.error("recorded feedback requires detection and archive labels")
    elif args.archive_run or args.source_row_map or args.archive_feedback is not None or args.recorded_detection_feedback:
        parser.error("Archive options require --teacher replay")
    if args.teacher == "live" and (args.max_requests is None or args.max_requests < 1):
        parser.error("live mode requires a positive --max-requests")
    return args


def run(args):
    import torch
    from threadpoolctl import threadpool_limits

    settings = config_for(args.dataset, args.config, verifier_override=args.verifier)
    args.seed = settings["runtime"]["seed"]
    args.device = settings["runtime"]["device"]
    args.threads = settings["runtime"]["threads"]
    args.max_requests = args.max_requests if args.max_requests is not None else 0
    dirty_path = (args.dirty or args.data_root / settings["dirty"]).resolve()
    dirty = Table.load(dirty_path)
    fields = fields_for(dirty, settings)
    converted, skipped = [], []
    pool_hash = None
    policy = None
    if args.rule_pool:
        pool_hash = digest(args.rule_pool.read_bytes())
        if args.legacy_rule_semantics == "archived":
            policy = {"archived": True, **settings.get("legacy_rule_import", {})}
        converted, skipped = load_pool(read_json(args.rule_pool), policy)
    rule_settings = {key: settings.get(key) for key in ("fields", "missing", "rules")}
    rule_settings["pool"] = pool_hash
    rule_settings["canonical_contracts"] = settings["detection"].get("canonical_contracts", [])
    rule_settings["contract_operations"] = contract_operations(settings)
    rules_sha256 = digest(encoded(rule_settings).encode("utf-8"))
    evidence_sha256 = evidence_fingerprint(settings, rules_sha256, args.dataset, policy)
    output = args.output.resolve()
    input_hash = digest((args.detections / "detections.csv").read_bytes()) if args.detections else None
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = output / "run.json"
    recorded_feedback = (args.teacher == "replay" and args.archive_feedback is not False and
                         (args.recorded_detection_feedback or
                          settings["detection"].get("feedback_schedule") == "archive"))
    manifest = {
        "version": __version__, "dataset": args.dataset, "mode": args.mode,
        "status": "running", "dirty": str(dirty_path), "dirty_sha256": dirty.sha256,
        "row_count": len(dirty.rows), "rule_pool_sha256": pool_hash, "rules_sha256": rules_sha256,
        "evidence_sha256": evidence_sha256, "accepted_legacy_detection_schema": args.accept_legacy_detections,
        "legacy_rule_semantics": args.legacy_rule_semantics, "legacy_rule_policy": policy,
        "detection_input_sha256": input_hash,
        "teacher": args.teacher, "api_request_budget": args.max_requests,
        "config_path": str(args.config.resolve()), "config_sha256": digest(args.config.read_bytes()),
        "recorded_detection_feedback": recorded_feedback,
        "seed": args.seed, "device": args.device, "threads": args.threads, "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name)
                     for name in ("numpy", "scikit-learn", "torch", "joblib", "threadpoolctl")},
        "code_sha256": {path.name: digest(path.read_bytes()) for path in sorted(Path(__file__).parent.glob("*.py"))},
        "config": settings, "skipped_legacy_rules": skipped, "converted_legacy_rules": len(converted),
        "requested_model": args.model or settings["teacher"].get("model"),
        "endpoint": args.endpoint or settings["teacher"].get("endpoint"),
    }
    write_json(manifest_path, manifest)
    teacher = None
    try:
        torch.set_num_threads(args.threads)
        torch.use_deterministic_algorithms(True)
        if args.teacher == "replay":
            source_path = args.data_root / settings["dirty"]
            source = dirty if source_path.resolve() == dirty_path else Table.load(source_path)
            teacher = ArchivedTeacher(args.archive_run, args.dataset, source, dirty, args.source_row_map,
                                      fields=fields, include_feedback=args.archive_feedback is not False,
                                      recorded_detection=recorded_feedback)
            manifest["supervision"] = teacher.provenance
            manifest["requested_model"] = teacher.provenance["model_requested"]
        else:
            teacher = Teacher(
                args.teacher, (args.cache or output / "teacher_cache").resolve(),
                manifest["requested_model"], manifest["endpoint"],
                args.max_requests, settings["teacher"]["timeout"], settings["teacher"]["attempts"],
            )
        with threadpool_limits(limits=args.threads):
            index = EvidenceIndex(dirty, fields, settings, converted)
            if index.missing.policies:
                write_csv(output / "missing_policy_audit.csv",
                          ["row_id", "column", "policy", "required", "suppressed_candidate_rules"],
                          index.missing.audit(index.pool))
            manifest["rule_operator_counts"] = Counter(rule.kind for rule in index.pool.rules)
            if args.mode == "correction":
                predictions = read_predictions(args.detections, dirty, rules_sha256,
                                               evidence_sha256=evidence_sha256,
                                               allow_legacy=args.accept_legacy_detections)
            else:
                predictions = detect(index, teacher, settings["detection"], manifest, output)
                write_predictions(output, dirty, predictions, rules_sha256, evidence_sha256=evidence_sha256)
            if args.mode != "detection":
                decisions = repair(index, predictions, teacher, settings["repair"], manifest, output)
                write_repaired(output, dirty, decisions)
        manifest.update(status="complete", candidate_count=len(predictions),
                        detected_count=sum(pred.error for pred in predictions))
        manifest["artifacts_sha256"] = {path.name: digest(path.read_bytes())
                                        for path in sorted(output.iterdir())
                                        if path.is_file() and path != manifest_path}
    except BaseException as error:
        manifest.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        if teacher:
            manifest.update(api_requests=teacher.requests, cache_hits=teacher.hits,
                            cache_misses=teacher.misses, served_models=sorted(teacher.served_models))
            if teacher.mode == "replay":
                manifest["supervision_counts"] = dict(teacher.counts)
        write_json(manifest_path, manifest)


def main():
    run(parse_args())
