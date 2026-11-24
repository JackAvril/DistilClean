"""Explicit field contracts backed by the same Dirty-only repair operators.

These are opt-in domain declarations, not rules inferred from reference labels.
An operator not applying says nothing about whether a cell is correct.
"""

from dataclasses import dataclass
import math
import re
from types import SimpleNamespace

from .repair_rules import direct_repairs


def contract_operations(config):
    declarations = config["detection"].get("canonical_contracts", [])
    available = config.get("repair", {}).get("direct_rules", [])
    result, columns = [], set()
    for declaration in declarations:
        if set(declaration) - {"operation_id", "teacher_conflicts"}:
            raise ValueError("Unknown canonical contract setting")
        if declaration.get("teacher_conflicts", "audit") not in {"audit", "quarantine"}:
            raise ValueError("Invalid canonical contract teacher conflict policy")
        found = [op for op in available if op["id"] == declaration["operation_id"]]
        if len(found) != 1:
            raise ValueError("Canonical contract must reference one existing repair operation")
        operation = found[0]
        if operation.get("scope") == "allowed":
            raise ValueError("Canonical contract cannot depend on detection output scope")
        if operation["column"] in columns:
            raise ValueError("Multiple canonical contracts target the same column")
        confidence = operation["confidence"]
        if not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Canonical contract confidence must be finite and in [0, 1]")
        if confidence < config["detection"]["threshold"]:
            raise ValueError("Canonical contract confidence is below the detection threshold")
        columns.add(operation["column"])
        result.append(operation)
    return result


@dataclass(frozen=True)
class ContractMatch:
    operation_id: str
    expected: str
    confidence: float
    teacher_conflicts: str


class CanonicalContracts:
    def __init__(self, table, fields, config):
        self.operations = contract_operations(config)
        for operation in self.operations:
            if operation["column"] not in fields:
                raise ValueError(f"Canonical contract column is absent: {operation['column']}")
        policies = {d["operation_id"]: d.get("teacher_conflicts", "audit")
                    for d in config["detection"].get("canonical_contracts", [])}
        index = SimpleNamespace(table=table, fields=fields)
        self.matches = {cell: ContractMatch(name, value, confidence, policies[name])
                        for cell, (value, name, confidence) in direct_repairs(index, self.operations).items()}

    def context(self, cell):
        match = self.matches.get(cell)
        if match is None:
            return {}
        return {"canonical_contract": {"operation_id": match.operation_id, "expected": match.expected,
                                       "confidence": match.confidence, "declaration": "explicit_field_contract"}}


class WeakEvidenceGuards:
    """A valid representation plus only weak anomaly signals cannot prove an error.

    Independent constraints (for example an FD not declared weak) block the guard.
    This is deliberately not a blanket assertion that well-formed values are correct.
    """

    def __init__(self, fields, declarations):
        self.fields = fields
        self.guards = {}
        names = set()
        for item in declarations:
            if set(item) - {"id", "column", "validator", "pattern", "weak_rule_types"}:
                raise ValueError("Unknown valid-value guard setting")
            column = item["column"]
            if column not in fields or not fields[column].focus or column in self.guards or item["id"] in names:
                raise ValueError("Invalid or repeated valid-value guard column/id")
            validator = item["validator"]
            if validator not in {"domain", "field", "pattern"}:
                raise ValueError("Unknown valid-value validator")
            if validator == "domain" and not fields[column].domain:
                raise ValueError("Domain guard requires a declared domain")
            if validator == "field" and fields[column].kind == "text":
                raise ValueError("Field guard requires a typed field")
            kinds = item["weak_rule_types"]
            if not kinds or any(not isinstance(kind, str) or not kind for kind in kinds):
                raise ValueError("Valid-value guard requires explicit weak rule types")
            pattern = re.compile(item["pattern"]) if validator == "pattern" else None
            self.guards[column] = (item, pattern)
            names.add(item["id"])

    def veto(self, cell, raw, signals):
        if cell.column not in self.guards or not signals:
            return ""
        item, pattern = self.guards[cell.column]
        field = self.fields[cell.column]
        if not field.key(raw) or field.issues(raw):
            return ""
        if item["validator"] == "pattern" and not pattern.fullmatch(raw.strip()):
            return ""
        kinds = {hit.rule.parameters.get("legacy", {}).get("rule_type", hit.rule.kind) for hit in signals}
        if kinds <= set(item["weak_rule_types"]):
            return item["id"]
        return ""
