"""Tests for _support utilities."""

import pytest

from agent_evolve.problem import ObjectiveSpec
from agent_evolve._support import (
    CandidateResult,
    evaluate_batch,
    format_search_space_description,
    parse_candidates,
    prettify_configuration,
    prettify_results,
    sample_failed_for_constraint,
)

OBJS = [ObjectiveSpec("value", "max"), ObjectiveSpec("weight", "min")]


# ------------------------------------------------------------------
# evaluate_batch
# ------------------------------------------------------------------

class _SimpleProblem:
    """Minimal problem for testing."""

    @property
    def objectives(self):
        return OBJS

    def validate(self, config):
        if config.get("x", -1) < 0:
            raise ValueError("x must be non-negative")
        return True

    def evaluate(self, config):
        return {"value": float(config["x"]), "weight": float(config["x"] * 2)}


class TestEvaluateBatch:
    def test_valid_candidates(self):
        p = _SimpleProblem()
        valid, failed, ordered = evaluate_batch(
            p, [{"x": 1}, {"x": 2}], OBJS, verbose=False
        )
        assert len(valid) == 2
        assert len(failed) == 0
        assert len(ordered) == 2
        assert valid[0].objectives["value"] == 1.0
        assert ordered[0] is valid[0] and ordered[1] is valid[1]

    def test_invalid_via_validate(self):
        p = _SimpleProblem()
        valid, failed, ordered = evaluate_batch(p, [{"x": -1}], OBJS, verbose=False)
        assert len(ordered) == 1
        assert len(valid) == 0
        assert len(failed) == 1
        assert "non-negative" in (failed[0].error_message or "")
        assert "ValueError" in (failed[0].error_message or "")

    def test_validate_returns_false_gives_actionable_hint(self):
        class FalseValidate:
            objectives = OBJS

            def validate(self, config):
                return False

            def evaluate(self, config):
                return {"value": 1.0, "weight": 1.0}

        valid, failed, _ = evaluate_batch(FalseValidate(), [{"x": 1}], OBJS, verbose=False)
        assert len(valid) == 0
        assert len(failed) == 1
        em = failed[0].error_message or ""
        assert "returned False" in em
        assert "ValueError" in em

    def test_exception_in_evaluate(self):
        class BadProblem:
            objectives = OBJS
            def evaluate(self, config):
                raise RuntimeError("boom")

        valid, failed, _ = evaluate_batch(BadProblem(), [{"x": 1}], OBJS, verbose=False)
        assert len(valid) == 0
        assert len(failed) == 1
        assert "boom" in (failed[0].error_message or "")
        assert "RuntimeError" in (failed[0].error_message or "")

    def test_no_validate_method(self):
        class NoValidate:
            objectives = OBJS
            def evaluate(self, config):
                return {"value": 1.0, "weight": 2.0}

        valid, failed, ordered = evaluate_batch(
            NoValidate(), [{"a": 1}], OBJS, verbose=False
        )
        assert len(valid) == 1
        assert len(ordered) == 1

    def test_ordered_matches_candidate_order(self):
        p = _SimpleProblem()
        valid, failed, ordered = evaluate_batch(
            p, [{"x": 1}, {"x": -1}, {"x": 2}], OBJS, verbose=False
        )
        assert len(ordered) == 3
        assert ordered[0].is_valid and ordered[1].is_valid is False
        assert ordered[2].is_valid
        assert ordered[0].objectives["value"] == 1.0
        assert ordered[2].objectives["value"] == 2.0


# ------------------------------------------------------------------
# parse_candidates
# ------------------------------------------------------------------

class TestParseCandidates:
    def test_dict_list(self):
        parsed, raw = parse_candidates([{"a": 1}, {"b": 2}], 2)
        assert parsed == [{"a": 1}, {"b": 2}]
        assert raw == [{"a": 1}, {"b": 2}]

    def test_json_strings(self):
        parsed, raw = parse_candidates(['{"a": 1}', '{"b": 2}'], 2)
        assert parsed == [{"a": 1}, {"b": 2}]
        assert raw[0] == '{"a": 1}'

    def test_non_list_returns_empty(self):
        assert parse_candidates("not a list", 1) == ([], [])

    def test_bad_json_becomes_empty_dict(self):
        parsed, raw = parse_candidates(["not json", {"a": 1}], 2)
        assert len(parsed) == 2
        assert parsed[0] == {}
        assert parsed[1] == {"a": 1}
        assert raw[0] == "not json"

    def test_dict_wrapper_uses_candidates_key(self):
        parsed, raw = parse_candidates(
            {"candidates": [{"x": 1}], "thought": "ok"},
            1,
            log_fn=lambda m: None,
        )
        assert parsed == [{"x": 1}]
        assert raw == [{"x": 1}]

    def test_json_string_parses_to_list(self):
        from agent_evolve._support import parse_llm_json_array

        s = '[{"selection": [0, 1]}, {"selection": [2]}]'
        assert parse_llm_json_array(s) == [{"selection": [0, 1]}, {"selection": [2]}]
        parsed, raw = parse_candidates(s, 2, log_fn=lambda m: None)
        assert parsed == [{"selection": [0, 1]}, {"selection": [2]}]


# ------------------------------------------------------------------
# prettify helpers
# ------------------------------------------------------------------

class TestPrettify:
    def test_prettify_configuration(self):
        s = prettify_configuration({"b": 2, "a": 1})
        assert '"a": 1' in s
        assert '"b": 2' in s

    def test_prettify_results_valid(self):
        cr = CandidateResult({"x": 1}, {"value": 10.0, "weight": 5.0}, is_valid=True)
        s = prettify_results([cr], OBJS)
        assert "value=10.0000" in s

    def test_prettify_results_invalid(self):
        cr = CandidateResult({}, {}, is_valid=False, error_message="bad config")
        s = prettify_results([cr], OBJS)
        assert "INVALID" in s
        assert "bad config" in s


# ------------------------------------------------------------------
# sample_failed_for_constraint
# ------------------------------------------------------------------

class TestSampleFailed:
    def test_all_latest_when_under_cap(self):
        f1 = CandidateResult({}, {}, False)
        f2 = CandidateResult({}, {}, False)
        result = sample_failed_for_constraint([f1, f2], [], 5)
        assert len(result) == 2

    def test_truncated_when_over_cap(self):
        failures = [CandidateResult({}, {}, False) for _ in range(10)]
        result = sample_failed_for_constraint(failures, [], 3)
        assert len(result) == 3


# ------------------------------------------------------------------
# format_search_space_description
# ------------------------------------------------------------------

class TestFormatSearchSpace:
    def test_basic(self):
        s = format_search_space_description(OBJS)
        assert "MULTI-OBJECTIVE" in s
        assert "value" in s
        assert "MAXIMIZE" in s

    def test_with_all_optional(self):
        s = format_search_space_description(
            OBJS,
            config_schema={"x": "int"},
            example_config={"x": 5},
            constraints="x > 0",
            problem_description="A test problem",
        )
        assert "CONFIGURATION SCHEMA" in s
        assert "EXAMPLE CONFIGURATION" in s
        assert "CONSTRAINTS" in s
        assert "PROBLEM DESCRIPTION" in s
