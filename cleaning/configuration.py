"""Load a complete external configuration without merging experiment presets."""
from __future__ import annotations

import json
import math
from pathlib import Path


DATASETS = ("hospital", "flights", "beers", "rayyan", "adult", "soccer")


def _object(properties, required=None):
    return {"type": "object", "properties": properties,
            "required": list(properties) if required is None else required}


def _integer(minimum):
    return {"type": "integer", "minimum": minimum}


def _number(minimum=0, maximum=None):
    result = {"type": "number", "minimum": minimum}
    if maximum is not None:
        result["maximum"] = maximum
    return result


STUDENT = _object({
    "hidden": {"type": "array", "items": _integer(1), "minItems": 1},
    "dropout": {"type": "number", "minimum": 0, "exclusiveMaximum": 1},
    "learning_rate": {"type": "number", "exclusiveMinimum": 0},
    "batch_size": _integer(1), "epochs": _integer(1), "weight_decay": _number(),
    "standardized_clip": {"anyOf": [{"type": "null"}, {"type": "number", "exclusiveMinimum": 0}]},
})
VERIFIER = _object({
    "minimum_samples": _integer(2), "n_estimators": _integer(1),
    "max_depth": {"anyOf": [{"type": "null"}, _integer(1)]},
    "class_weight": {"enum": [None, "balanced", "balanced_subsample"]},
    "n_jobs": {"anyOf": [_integer(1), {"enum": [-1]}]},
})
DETECTION = _object({
    "rare_max_count": _integer(0), "rare_pattern_ratio": _number(0, 1),
    "neighbor_pool": _integer(1), "neighbors": _integer(1),
    "teacher_budget": _integer(0), "feedback_budget": _integer(0),
    "cluster_limit": _integer(1), "propagation_threshold": _number(0, 1),
    "pseudo_negative_budget": _integer(0),
    "calibration_fraction": {"type": "number", "minimum": 0, "exclusiveMaximum": 1},
    "threshold": _number(0, 1), "verifier_threshold": _number(0, 1), "rounds": _integer(1),
    "feedback_student": {"enum": ["update", "freeze"]},
    "initial_replay": {"enum": ["resample", "archive"]}, "student": STUDENT, "verifier": VERIFIER,
})
REPAIR = _object({
    "max_candidates": _integer(1), "frequency_candidates": _integer(0),
    "teacher_alternatives": _integer(1), "teacher_budget": _integer(0), "feedback_budget": _integer(0),
    "cluster_limit": _integer(1), "margin": _number(), "rounds": _integer(1),
    "expand_consensus": {"type": "boolean"}, "expansion_confidence": _number(0, 1), "student": STUDENT,
})
CONFIG_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "DistilClean external experiment configuration",
    **_object({
        "dataset": {"enum": list(DATASETS)}, "dirty": {"type": "string", "minLength": 1},
        "missing": {"type": "array", "items": {"type": "string"}},
        "fields": {"type": "object"}, "rules": {"type": "array", "items": {"type": "object"}},
        "detection": DETECTION, "repair": REPAIR,
        "teacher": _object({"timeout": {"type": "number", "exclusiveMinimum": 0}, "attempts": _integer(1)}),
        "runtime": _object({"seed": {"type": "integer", "minimum": 0, "maximum": 2**32 - 2},
                            "device": {"enum": ["cpu", "cuda"]}, "threads": _integer(1)}),
    }),
}


def _validate(value, schema, location="config"):
    if "anyOf" in schema:
        for option in schema["anyOf"]:
            try:
                _validate(value, option, location)
                return
            except ValueError:
                pass
        raise ValueError(f"{location}: value does not match an allowed type or range")
    if "enum" in schema and not any(type(value) is type(choice) and value == choice for choice in schema["enum"]):
        raise ValueError(f"{location}: select an explicitly allowed value")
    kind = schema.get("type")
    matches = {"object": isinstance(value, dict), "array": isinstance(value, list),
               "string": isinstance(value, str), "boolean": type(value) is bool,
               "integer": type(value) is int, "number": type(value) in (int, float), "null": value is None}
    if kind is not None and not matches[kind]:
        raise ValueError(f"{location}: expected {kind}")
    if type(value) in (int, float):
        if not math.isfinite(value):
            raise ValueError(f"{location}: expected a finite number")
        checks = {"minimum": lambda x: value >= x, "maximum": lambda x: value <= x,
                  "exclusiveMinimum": lambda x: value > x, "exclusiveMaximum": lambda x: value < x}
        if any(not check(schema[key]) for key, check in checks.items() if key in schema):
            raise ValueError(f"{location}: value is outside its allowed range")
    if kind == "object":
        missing = [key for key in schema.get("required", []) if key not in value]
        if missing:
            raise ValueError(f"{location}: explicitly configure {', '.join(missing)}")
        for key, subschema in schema.get("properties", {}).items():
            if key in value:
                _validate(value[key], subschema, f"{location}.{key}")
    elif kind == "array":
        if len(value) < schema.get("minItems", 0):
            raise ValueError(f"{location}: expected a nonempty array")
        for index, item in enumerate(value):
            _validate(item, schema["items"], f"{location}[{index}]")
    elif kind == "string" and len(value) < schema.get("minLength", 0):
        raise ValueError(f"{location}: expected a nonempty string")


def validate_configuration(value, dataset):
    _validate(value, CONFIG_SCHEMA)
    if value["dataset"] != dataset:
        raise ValueError("Configuration dataset does not match --dataset")
    if any(key in value for key in ("clean", "evaluation", "defaults")):
        raise ValueError("Reference tables, evaluation and preset merging are not runtime inputs")
    if any(key in value["teacher"] for key in ("api_key", "token", "password")):
        raise ValueError("Supply teacher credentials through environment variables")
    return value


def config_for(dataset, configuration_path):
    """Require one complete JSON configuration; never read bundled dataset defaults."""
    path = Path(configuration_path)
    with path.open(encoding="utf-8-sig") as handle:
        value = json.load(handle)
    return validate_configuration(value, dataset)
