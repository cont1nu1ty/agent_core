"""Safe corrections for actual business validators without echoing input."""

from __future__ import annotations

from typing import ClassVar
import unittest

from pydantic import BaseModel, ConfigDict, model_validator
from pydantic_core import PydanticCustomError

from agent_core.runtime.cancellation import CancellationToken
from agent_core.tools import ToolExecutor, ToolRegistry, ToolSpec
from agent_core.tools.validation import ToolValidationHint
from agent_core_tests.test_runtime import MinimalToolRuntime


class DeclaredInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    tool_validation_hints: ClassVar[dict[str, ToolValidationHint]] = {
        "duplicate_items": ToolValidationHint(("items",), "Remove duplicate items before retrying."),
    }
    items: list[str]

    @model_validator(mode="after")
    def unique_items(self):
        if len(set(self.items)) != len(self.items):
            raise PydanticCustomError("duplicate_items", "PRIVATE exception detail {value}", {"value": self.items[0]})
        return self


class ValidationFeedbackTest(unittest.TestCase):
    def execute(self, schema, arguments):
        calls = []

        def handler(parsed, runtime):
            calls.append(parsed)
            return "valid"

        definition = ToolSpec("validate_request", "Validate a request", schema, handler)
        runtime = MinimalToolRuntime(CancellationToken())
        result = ToolExecutor(registry=ToolRegistry([definition])).execute(
            call_id="validation-1", tool_name=definition.name, arguments=arguments, runtime=runtime)
        return result, calls, runtime

    def test_declared_hint_is_actionable_without_error_context_or_input(self):
        result, calls, runtime = self.execute(DeclaredInput, {"items": ["PRIVATE-value", "PRIVATE-value"]})
        self.assertFalse(result.succeeded)
        self.assertEqual(result.data["errorCode"], "invalid_tool_arguments")
        self.assertEqual(result.data["details"]["errors"], [{
            "field": "items", "fields": ["items"], "code": "duplicate_items",
            "hint": "Remove duplicate items before retrying.", "constraints": {"type": "array"},
        }])
        self.assertNotIn("PRIVATE", result.content)
        self.assertEqual(calls, [])
        self.assertEqual(runtime.execution.logical_tool_calls, 1)

    def test_unknown_custom_error_code_cannot_echo_a_value_disguised_as_code(self):
        class UnknownInput(BaseModel):
            value: str

            @model_validator(mode="after")
            def reject(self):
                raise PydanticCustomError(self.value, "PRIVATE exception {value}", {"value": self.value})

        result, calls, _ = self.execute(UnknownInput, {"value": "private_secret_012"})
        self.assertEqual(result.data["details"]["errors"][0]["code"], "invalid")
        self.assertNotIn("private_secret_012", result.content)
        self.assertNotIn("PRIVATE", result.content)
        self.assertEqual(calls, [])

    def test_unregistered_hint_field_fails_closed(self):
        class BrokenDeclaration(DeclaredInput):
            tool_validation_hints: ClassVar[dict[str, ToolValidationHint]] = {
                "duplicate_items": ToolValidationHint(("internal_private_field",), "PRIVATE trusted typo"),
            }

        result, _, _ = self.execute(BrokenDeclaration, {"items": ["PRIVATE", "PRIVATE"]})
        error = result.data["details"]["errors"][0]
        self.assertEqual(error["code"], "invalid")
        self.assertNotIn("hint", error)
        self.assertNotIn("internal_private_field", result.content)
        self.assertNotIn("PRIVATE", result.content)

    def test_plain_value_error_keeps_safe_standard_code(self):
        class PlainInput(BaseModel):
            value: str

            @model_validator(mode="after")
            def reject(self):
                raise ValueError("PRIVATE exception details " + self.value)

        result, _, _ = self.execute(PlainInput, {"value": "PRIVATE-value"})
        self.assertEqual(result.data["details"]["errors"][0]["code"], "value_error")
        self.assertNotIn("PRIVATE", result.content)

    def test_hint_metadata_is_not_a_model_argument_and_corrected_input_executes(self):
        self.assertEqual(set(DeclaredInput.model_json_schema()["properties"]), {"items"})
        invalid, calls, _ = self.execute(DeclaredInput, {
            "items": ["one"], "tool_validation_hints": {"duplicate_items": "PRIVATE override"},
        })
        self.assertEqual(invalid.data["details"]["errors"][0], {"field": "<field>", "code": "extra_forbidden"})
        self.assertNotIn("PRIVATE", invalid.content)
        self.assertEqual(calls, [])
        valid, calls, _ = self.execute(DeclaredInput, {"items": ["one", "two"]})
        self.assertTrue(valid.succeeded)
        self.assertEqual(len(calls), 1)




if __name__ == "__main__":
    unittest.main()
