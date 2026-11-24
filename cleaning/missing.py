"""Explicit field missingness semantics, evaluated against Dirty only."""


class MissingSemantics:
    def __init__(self, table, fields, config):
        self.table, self.fields = table, fields
        self.policies = {}
        for column in fields:
            spec = config.get("fields", {}).get(column, {})
            policy = spec.get("missing_policy")
            if policy is None:
                continue
            if not isinstance(policy, dict) or policy.get("mode") not in {"allowed", "required", "conditional"}:
                raise ValueError(f"Invalid missing policy for {column}")
            mode = policy["mode"]
            if fields[column].required and mode != "required":
                raise ValueError(f"Missing policy contradicts required schema: {column}")
            conditions = policy.get("required_when", [])
            if mode == "conditional" and (not isinstance(conditions, list) or not conditions):
                raise ValueError(f"Conditional missing policy needs conditions: {column}")
            if mode != "conditional" and conditions:
                raise ValueError(f"Unexpected missing conditions: {column}")
            for condition in conditions:
                other = condition.get("column")
                if other not in fields or other == column:
                    raise ValueError(f"Invalid missing context column: {column}")
                if ("present" in condition) == ("values" in condition):
                    raise ValueError(f"Missing condition needs present or values: {column}")
                if "present" in condition and type(condition["present"]) is not bool:
                    raise ValueError(f"Missing presence condition must be boolean: {column}")
                if "values" in condition and (not isinstance(condition["values"], list) or not condition["values"]):
                    raise ValueError(f"Missing value condition needs values: {column}")
            self.policies[column] = policy

    def requirement(self, cell):
        policy = self.policies.get(cell.column)
        if policy is None:
            return None  # Inherit the existing schema/rule behavior exactly.
        if policy["mode"] != "conditional":
            return policy["mode"] == "required"
        row = self.table.rows[cell.row_id]
        for condition in policy["required_when"]:
            other = condition["column"]
            field = self.fields[other]
            key = field.key(row[self.table.index[other]])
            if "present" in condition:
                matches = bool(key) == condition["present"]
            else:
                matches = key in {field.key(value) for value in condition["values"]}
            if not matches:
                return False
        return True

    def suppresses(self, cell, rule):
        """Allowed missingness overrules mined candidate hints, never a hard rule."""
        return (self.requirement(cell) is False and rule.kind in {"required", "paired_missing"}
                and rule.parameters.get("legacy", {}).get("usage_role") == "candidate_generation_rule")

    def audit(self, pool):
        for row_id, row in enumerate(self.table.rows):
            for column, policy in self.policies.items():
                if self.fields[column].key(row[self.table.index[column]]):
                    continue
                from .data import Cell
                cell = Cell(row_id, column)
                suppressed = [rule.id for rule in pool.by_target[column] if self.suppresses(cell, rule)]
                yield (row_id, column, policy["mode"], int(self.requirement(cell)), "|".join(suppressed))
