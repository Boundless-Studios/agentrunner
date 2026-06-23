"""Tests for the JSON sanitiser used to recover malformed LLM JSON output."""
import json

import pytest

from agentrunner.json_sanitizer import parse_json_safely, sanitize_json_string


class TestMarkdownFenceStripping:
    """Open-source LLMs (Kimi K2.6, K2.5) routinely wrap structured-output
    JSON in a ``` ```json ... ``` ``` markdown block despite an explicit
    "no markdown formatting" instruction. The sanitiser must recover."""

    def test_kimi_style_json_fence_round_trips(self):
        raw = (
            "```json\n"
            '{"campaign_id": "vell-tower-amnesia-001",'
            ' "title": "The Mnemosyne Labyrinth"}\n'
            "```"
        )
        out = sanitize_json_string(raw)
        parsed = json.loads(out)
        assert parsed["campaign_id"] == "vell-tower-amnesia-001"
        assert parsed["title"] == "The Mnemosyne Labyrinth"

    def test_bare_triple_backtick_fence_round_trips(self):
        raw = "```\n" '{"a": 1}\n' "```"
        assert json.loads(sanitize_json_string(raw))["a"] == 1

    def test_parse_json_safely_recovers_from_markdown_fence(self):
        raw = "```json\n" '{"x": 42}\n' "```"
        assert parse_json_safely(raw) == {"x": 42}

    def test_plain_json_is_unaffected(self):
        raw = '{"a": 1, "b": [2, 3]}'
        assert json.loads(sanitize_json_string(raw)) == {"a": 1, "b": [2, 3]}


class TestTrailingCommaRepair:
    """Open-weight models (notably Nemotron-120B) emit structurally-complete JSON
    with trailing commas before a closing brace/bracket, which strict json.loads
    rejects with 'Expecting property name enclosed in double quotes'. The
    sanitiser (and thus the agents-SDK validate_json patch) must recover."""

    def test_trailing_comma_before_object_close(self):
        raw = '{"a": {"x": "for the zenith.",}, "b": 2}'
        with pytest.raises(json.JSONDecodeError):
            json.loads(raw)
        assert json.loads(sanitize_json_string(raw)) == {"a": {"x": "for the zenith."}, "b": 2}

    def test_trailing_comma_before_array_close(self):
        raw = '{"items": [1, 2, 3,], "ok": true}'
        with pytest.raises(json.JSONDecodeError):
            json.loads(raw)
        assert json.loads(sanitize_json_string(raw)) == {"items": [1, 2, 3], "ok": True}

    def test_trailing_comma_with_newline_indent(self):
        # The exact shape captured from a real Nemotron-120B campaign generation:
        # `"...zenith.",\n        },\n        {`
        raw = '{"acts": [{"goal": "for the zenith.",\n        },\n        {"goal": "next"}]}'
        with pytest.raises(json.JSONDecodeError):
            json.loads(raw)
        parsed = json.loads(sanitize_json_string(raw))
        assert parsed["acts"][0]["goal"] == "for the zenith."
        assert parsed["acts"][1]["goal"] == "next"

    def test_nested_trailing_commas(self):
        raw = '{"a": [{"b": 1,}, {"c": 2,},],}'
        assert json.loads(sanitize_json_string(raw)) == {"a": [{"b": 1}, {"c": 2}]}

    def test_comma_inside_string_value_preserved(self):
        # Regression: the structural trailing-comma strip must NOT touch a comma
        # inside string content like "choose A,] then B" while still fixing the
        # structural trailing commas (review 3371840721).
        raw = '{"text": "choose A,] then B", "items": [1, 2,],}'
        parsed = json.loads(sanitize_json_string(raw))
        assert parsed["text"] == "choose A,] then B"
        assert parsed["items"] == [1, 2]

    def test_structural_strip_is_string_aware_in_isolation(self):
        # The structural-comma helper itself must honour string + escape state.
        from agentrunner.json_sanitizer import _strip_structural_trailing_commas
        assert _strip_structural_trailing_commas('{"q": "a,] b",}') == '{"q": "a,] b"}'
        assert _strip_structural_trailing_commas('{"q": "x\\",}y",}') == '{"q": "x\\",}y"}'
