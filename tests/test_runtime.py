import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cleaning.configuration import CONFIG_SCHEMA, config_for, validate_configuration
from cleaning.data import Cell, Table, write_json
from cleaning.detection import Prediction
from cleaning.pipeline import parse_args
from cleaning.repair import Repair
from cleaning.results import read_predictions, write_predictions, write_repaired


def synthetic_value(schema):
    """Generate type-valid test values; these are not experiment presets."""
    if "enum" in schema:
        return schema["enum"][0]
    if "anyOf" in schema:
        return synthetic_value(schema["anyOf"][0])
    kind = schema["type"]
    if kind == "object":
        return {key: synthetic_value(schema["properties"][key]) for key in schema.get("required", [])}
    if kind == "array":
        return [synthetic_value(schema["items"]) for _ in range(schema.get("minItems", 0))]
    if kind == "string":
        return "synthetic"
    if kind == "boolean":
        return False
    if kind == "null":
        return None
    return schema["exclusiveMinimum"] + 1 if "exclusiveMinimum" in schema else schema.get("minimum", 0)


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.config = synthetic_value(CONFIG_SCHEMA)

    def test_schema_contains_no_default_values(self):
        def visit(value):
            if isinstance(value, dict):
                self.assertNotIn("default", value)
                for item in value.values():
                    visit(item)
            elif isinstance(value, list):
                for item in value:
                    visit(item)
        visit(CONFIG_SCHEMA)

    def test_external_configuration_is_required(self):
        with self.assertRaises(TypeError):
            config_for("hospital")

    def test_complete_configuration_is_loaded_without_merging(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "experiment.json"
            write_json(path, self.config)
            self.assertEqual(config_for("hospital", path), self.config)

    def test_missing_training_parameter_is_rejected(self):
        del self.config["detection"]["student"]["learning_rate"]
        with self.assertRaisesRegex(ValueError, "learning_rate"):
            validate_configuration(self.config, "hospital")

    def test_missing_runtime_parameter_is_rejected(self):
        del self.config["runtime"]["seed"]
        with self.assertRaisesRegex(ValueError, "seed"):
            validate_configuration(self.config, "hospital")

    def test_missing_verifier_parameter_is_rejected(self):
        del self.config["detection"]["verifier"]["n_estimators"]
        with self.assertRaisesRegex(ValueError, "n_estimators"):
            validate_configuration(self.config, "hospital")

    def test_dataset_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "dataset does not match"):
            validate_configuration(self.config, "soccer")

    def test_boolean_and_nonfinite_hyperparameters_are_rejected(self):
        for value in (True, float("nan"), float("inf"), -1):
            config = copy.deepcopy(self.config)
            config["detection"]["student"]["learning_rate"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_configuration(config, "hospital")

    def test_reference_tables_and_inline_credentials_are_rejected(self):
        config = copy.deepcopy(self.config)
        config["clean"] = "reference.csv"
        with self.assertRaises(ValueError):
            validate_configuration(config, "hospital")
        config = copy.deepcopy(self.config)
        config["teacher"]["api_key"] = "not-a-real-key"
        with self.assertRaises(ValueError):
            validate_configuration(config, "hospital")

    def test_removed_cli_options_are_rejected(self):
        arguments = ["cleaning", "all", "--dataset", "hospital", "--data-root", "unused",
                     "--config", "external.json", "--rule-pool", "external-pool.json",
                     "--legacy-rule-semantics", "archived", "--teacher", "fixture", "--output", "unused"]
        for option in ("--quick", "--evaluate", "--clean"):
            with self.subTest(option=option), patch("sys.argv", arguments + [option]), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as result:
                    parse_args()
                self.assertEqual(result.exception.code, 2)

    def test_rule_generation_mode_is_rejected(self):
        with patch("sys.argv", ["cleaning", "rules"]), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as result:
                parse_args()
        self.assertEqual(result.exception.code, 2)

    def test_live_requests_need_an_explicit_cap(self):
        arguments = ["cleaning", "all", "--dataset", "hospital", "--data-root", "unused",
                     "--config", "external.json", "--rule-pool", "external-pool.json",
                     "--legacy-rule-semantics", "archived", "--teacher", "live", "--output", "unused"]
        with patch("sys.argv", arguments), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as result:
                parse_args()
        self.assertEqual(result.exception.code, 2)


class OutputTests(unittest.TestCase):
    def test_detection_output_preserves_coordinates_and_checks_input(self):
        table = Table(("value",), [("odd",), ("stable",)], "synthetic-dirty")
        prediction = Prediction(Cell(0, "value"), "odd", .75, .75, True)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            write_predictions(output, table, [prediction], "synthetic-rules", evidence_sha256="synthetic-evidence")
            self.assertEqual(read_predictions(output, table, "synthetic-rules", evidence_sha256="synthetic-evidence"), [prediction])
            with self.assertRaises(ValueError):
                read_predictions(output, table, "other-rules")

    def test_repair_writes_only_the_decided_cells(self):
        table = Table(("value",), [("odd",), ("stable",)], "synthetic-dirty")
        repair = Repair(Cell(0, "value"), "odd", "fixed", "fixed", 1, 1, True, "synthetic", "rule")
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            write_repaired(output, table, [repair])
            self.assertEqual(Table.load(output / "repaired.csv").rows, [("fixed",), ("stable",)])


if __name__ == "__main__":
    unittest.main()
