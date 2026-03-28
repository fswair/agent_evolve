"""Internal helpers used by evolve.kedi's embedded Python blocks.

Public API users should not import from this module directly.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel

from agent_evolve.problem import ObjectiveSpec
from agent_evolve.results import (
    Candidate,
    SearchResult,
    compute_pareto_front,
    dominates,
    select_minimax_rank,
    sort_by_minimax_rank,
)


# ------------------------------------------------------------------
# Internal candidate tracking
# ------------------------------------------------------------------

@dataclass
class CandidateResult:
    """Intermediate evaluation record (richer than public Candidate)."""

    configuration: Dict[str, Any]
    objectives: Dict[str, float]
    is_valid: bool
    error_message: Optional[str] = None
    insight: str = ""
    #: Original element from the LLM ``candidates`` list (before/while parsing).
    raw_llm_element: Optional[Any] = None


# ------------------------------------------------------------------
# Evaluate a batch of candidates against a Problem
# ------------------------------------------------------------------

INVALID_PENALTY: float = 1e18

_VALIDATION_FALSE_HINT = (
    "validate() returned False without a reason. "
    "Raise ValueError('...') from validate() describing what is wrong and how to fix it "
    "(unknown keys, wrong types, out-of-range values, invalid combinations)."
)


def format_optimizer_error(exc: BaseException) -> str:
    """Format an exception for regeneration prompts so the model gets a concrete fix hint."""
    name = type(exc).__name__
    msg = str(exc).strip()
    if msg:
        return f"{name}: {msg}"
    return (
        f"{name} was raised with no message. "
        "Raise ValueError('clear explanation of what failed and valid alternatives') instead."
    )


def evaluate_batch(
    problem: Any,
    candidates: List[Dict[str, Any]],
    objectives: Sequence[ObjectiveSpec],
    *,
    raw_llm_elements: Optional[List[Any]] = None,
    verbose: bool = True,
    log_fn: Callable[[str], None] = lambda m: print(m, flush=True),
) -> Tuple[List[CandidateResult], List[CandidateResult], List[CandidateResult]]:
    """Evaluate *candidates* against *problem*.

    If *raw_llm_elements* is set (parallel to *candidates*), each result keeps the
    corresponding list element so failure prompts show what the model actually sent.

    Returns ``(valid, failed, ordered)`` where *ordered* has one :class:`CandidateResult`
    per input candidate, in the same order as *candidates*.
    """
    valid: List[CandidateResult] = []
    failed: List[CandidateResult] = []
    ordered: List[CandidateResult] = []
    has_validate = hasattr(problem, "validate")

    def _raw(idx: int) -> Optional[Any]:
        if not raw_llm_elements or idx >= len(raw_llm_elements):
            return None
        return raw_llm_elements[idx]

    for idx, config in enumerate(candidates):
        try:
            if verbose:
                log_fn(f"    Candidate {idx + 1}: {prettify_configuration(config)[:200]}...")

            if has_validate:
                try:
                    ok = problem.validate(config)
                except Exception as exc:
                    cr = _make_failure_result(
                        config,
                        objectives,
                        format_optimizer_error(exc),
                        verbose,
                        log_fn,
                        raw_llm_element=_raw(idx),
                    )
                    failed.append(cr)
                    ordered.append(cr)
                    continue
                if not ok:
                    cr = _make_failure_result(
                        config,
                        objectives,
                        _VALIDATION_FALSE_HINT,
                        verbose,
                        log_fn,
                        raw_llm_element=_raw(idx),
                    )
                    failed.append(cr)
                    ordered.append(cr)
                    continue

            obj = problem.evaluate(config)
            if verbose:
                log_fn(f"    -> VALID: {obj}")
            cr = CandidateResult(
                configuration=config,
                objectives=obj,
                is_valid=True,
                raw_llm_element=_raw(idx),
            )
            valid.append(cr)
            ordered.append(cr)

        except Exception as exc:
            cr = _make_failure_result(
                config,
                objectives,
                format_optimizer_error(exc),
                verbose,
                log_fn,
                raw_llm_element=_raw(idx),
            )
            failed.append(cr)
            ordered.append(cr)

    return valid, failed, ordered


def _make_failure_result(
    config: Dict[str, Any],
    objectives: Sequence[ObjectiveSpec],
    message: str,
    verbose: bool,
    log_fn: Callable[[str], None],
    *,
    raw_llm_element: Optional[Any] = None,
) -> CandidateResult:
    if verbose:
        log_fn(f"    -> FAILED: {message}")
    return CandidateResult(
        configuration=config,
        objectives={
            s.name: (0.0 if s.goal == "max" else INVALID_PENALTY) for s in objectives
        },
        is_valid=False,
        error_message=message,
        raw_llm_element=raw_llm_element,
    )


# ------------------------------------------------------------------
# Formatting helpers (consumed by LLM prompts)
# ------------------------------------------------------------------

def prettify_configuration(config: Dict[str, Any], indent: int = 2) -> str:
    return json.dumps(config, indent=indent, sort_keys=True)


def dump_raw_llm_element(obj: Any, *, max_len: int = 12_000) -> str:
    """Serialize a raw LLM list element for logs and failure prompts."""
    if obj is None:
        return "(none)"
    try:
        s = json.dumps(obj, indent=2, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        s = repr(obj)
    if len(s) > max_len:
        return s[:max_len] + f"\n... [truncated, {len(s)} chars total]"
    return s


def prettify_objectives(objectives: Sequence[ObjectiveSpec]) -> str:
    lines = ["OBJECTIVES:", "=" * 60]
    for spec in objectives:
        desc = "higher is better" if spec.goal == "max" else "lower is better"
        lines.append(f"  - {spec.name}: {desc}")
    return "\n".join(lines)


def prettify_results(
    results: List[CandidateResult],
    objectives: Sequence[ObjectiveSpec],
) -> str:
    lines: List[str] = []
    for i, r in enumerate(results, 1):
        lines.append(f"--- Candidate {i} ---")
        lines.append(f"Configuration: {prettify_configuration(r.configuration)}")
        if getattr(r, "raw_llm_element", None) is not None:
            lines.append(
                "Raw LLM element (exact item from the model's candidates list): "
                + dump_raw_llm_element(r.raw_llm_element)
            )
        if r.is_valid:
            parts = []
            for spec in objectives:
                val = r.objectives.get(spec.name, 0.0)
                arrow = "\u2191" if spec.goal == "max" else "\u2193"
                parts.append(f"{spec.name}={val:.4f}{arrow}")
            lines.append(f"Objectives: {', '.join(parts)}")
        else:
            lines.append("Status: INVALID")
            if r.error_message:
                lines.append(f"Error: {r.error_message}")
        if r.insight:
            lines.append(f"Insight: {r.insight}")
        lines.append("")
    return "\n".join(lines)


# ------------------------------------------------------------------
# Search-space description for LLM context
# ------------------------------------------------------------------

def format_search_space_description(
    objectives: Sequence[ObjectiveSpec],
    *,
    config_schema: Optional[Dict[str, Any]] = None,
    example_config: Optional[Dict[str, Any]] = None,
    constraints: Optional[str] = None,
    problem_description: Optional[str] = None,
) -> str:
    lines: List[str] = []
    lines.append("=" * 70)
    lines.append("MULTI-OBJECTIVE OPTIMIZATION PROBLEM")
    lines.append("=" * 70)
    lines.append("")
    lines.append("OBJECTIVES:")
    for spec in objectives:
        desc = "MAXIMIZE (higher is better)" if spec.goal == "max" else "MINIMIZE (lower is better)"
        lines.append(f"  \u2022 {spec.name}: {desc}")
    lines.append("")

    if problem_description:
        lines.append("PROBLEM DESCRIPTION:")
        lines.append(problem_description)
        lines.append("")

    if config_schema:
        lines.append("CONFIGURATION SCHEMA:")
        lines.append(prettify_configuration(config_schema))
        lines.append("")

    if example_config:
        lines.append("EXAMPLE CONFIGURATION:")
        lines.append(prettify_configuration(example_config))
        lines.append("")

    if constraints:
        lines.append("CONSTRAINTS:")
        lines.append(constraints)
        lines.append("")

    return "\n".join(lines)


# ------------------------------------------------------------------
# Performance statistics
# ------------------------------------------------------------------

def compute_performance_stats(
    valid_results: List[CandidateResult],
    objectives: Sequence[ObjectiveSpec],
) -> Optional[Dict[str, Any]]:
    """Compute best/worst per objective and top Pareto candidates."""
    if not valid_results:
        return None

    stats: Dict[str, Any] = {}

    for spec in objectives:
        key = spec.name
        if spec.goal == "max":
            best = max(valid_results, key=lambda r: r.objectives.get(key, 0.0))
            worst = min(valid_results, key=lambda r: r.objectives.get(key, 0.0))
        else:
            best = min(valid_results, key=lambda r: r.objectives.get(key, float("inf")))
            worst = max(valid_results, key=lambda r: r.objectives.get(key, 0.0))
        stats[f"best_{key}"] = best
        stats[f"worst_{key}"] = worst

    candidates = [result_to_candidate(r) for r in valid_results]
    pareto_candidates = compute_pareto_front(candidates, objectives)
    sorted_pareto = sort_by_minimax_rank(pareto_candidates, objectives)
    pareto_results = [candidate_to_result(c) for c in sorted_pareto]
    stats["top_3_pareto"] = pareto_results[:3]
    stats["pareto_front"] = pareto_results
    stats["pareto_size"] = len(pareto_results)

    return stats


# ------------------------------------------------------------------
# Failure sampling for constraint learning
# ------------------------------------------------------------------

def sample_failed_for_constraint(
    latest_failed: List[CandidateResult],
    all_previous_failed: List[CandidateResult],
    max_examples: int,
) -> List[CandidateResult]:
    """Sample failures for constraint-instruction generation.

    Always includes latest failures; fills remaining slots with random
    previous failures.
    """
    sampled = list(latest_failed)
    if len(sampled) >= max_examples:
        return sampled[:max_examples]

    remaining = max_examples - len(sampled)
    latest_ids = {id(r) for r in latest_failed}
    previous = [r for r in all_previous_failed if id(r) not in latest_ids]
    if previous and remaining > 0:
        sampled.extend(random.sample(previous, min(remaining, len(previous))))
    return sampled


# ------------------------------------------------------------------
# Conversions between CandidateResult and public Candidate
# ------------------------------------------------------------------

def result_to_candidate(
    result: CandidateResult,
    metadata: Optional[Dict[str, Any]] = None,
) -> Candidate[Dict[str, Any]]:
    return Candidate(
        configuration=result.configuration,
        objectives=result.objectives,
        metadata=metadata or {"is_pareto": False},
    )


def candidate_to_result(candidate: Candidate[Dict[str, Any]]) -> CandidateResult:
    return CandidateResult(
        configuration=candidate.configuration,
        objectives=candidate.objectives,
        is_valid=True,
    )


# ------------------------------------------------------------------
# Parse LLM candidate output
# ------------------------------------------------------------------

def parse_llm_json_array(s: str) -> list[Any]:
    """Parse a JSON array (or single object) from an LLM ``str`` field.

    Pydantic structured output with ``list[dict]`` often degrades to ``[{}, {}, ...]``.
    Returning the array as **one JSON string** avoids that; this function parses it
    and strips optional markdown fences.
    """
    import re

    s = (s or "").strip()
    if not s:
        raise ValueError("Empty candidates JSON string")
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```\s*$", "", s)
    data = json.loads(s)
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data
    raise ValueError(f"Candidates JSON must be an array or object, got {type(data).__name__}")


def parse_candidates(
    candidates: Any,
    expected_count: int,
    log_fn: Callable[[str], None] = lambda m: print(m, flush=True),
) -> Tuple[List[Dict[str, Any]], List[Any]]:
    """Normalise LLM output into configuration dicts.

    Returns ``(parsed_configs, raw_elements)`` with one raw element per input list
    item (same order), so failures can show what the model actually returned even
    when the parsed dict is ``{}`` or wrong.
    """
    if isinstance(candidates, str):
        try:
            candidates = parse_llm_json_array(candidates)
        except Exception as exc:
            log_fn(f"Warning: could not parse candidates as JSON array string: {exc}")
            return [], []

    if isinstance(candidates, dict):
        inner = candidates.get("candidates")
        if isinstance(inner, list):
            log_fn(
                "Note: LLM returned an object with a 'candidates' list; "
                "using that list (not the outer dict)."
            )
            candidates = inner
        else:
            log_fn(
                f"Warning: LLM returned a dict without a 'candidates' list "
                f"(keys: {list(candidates.keys())}). Expected a JSON list of configs."
            )
            return [], []

    if not isinstance(candidates, list):
        log_fn(f"Warning: LLM returned non-list candidates: {type(candidates)}")
        return [], []

    parsed: List[Dict[str, Any]] = []
    raw_elements: List[Any] = []
    for c in candidates:
        raw_elements.append(c)
        if isinstance(c, BaseModel):
            parsed.append(c.model_dump())
        elif isinstance(c, dict):
            parsed.append(c)
        elif isinstance(c, str):
            try:
                parsed.append(json.loads(c))
            except Exception:
                log_fn(f"Warning: Could not parse candidate string: {c[:100]}")
                parsed.append({})
        else:
            log_fn(
                f"Warning: candidate element has unexpected type {type(c).__name__}: "
                f"{repr(c)[:200]}"
            )
            parsed.append({})

    if len(parsed) != expected_count:
        log_fn(f"Warning: Expected {expected_count} candidates, got {len(parsed)}")
    return parsed, raw_elements


# ------------------------------------------------------------------
# Build final SearchResult from internal bookkeeping
# ------------------------------------------------------------------

def build_search_result(
    all_valid: List[CandidateResult],
    all_candidates_meta: List[Tuple[CandidateResult, Dict[str, Any]]],
    objectives: Sequence[ObjectiveSpec],
    history: List[Dict[str, Any]],
    best_per_generation: Optional[List[Candidate[Dict[str, Any]]]] = None,
) -> SearchResult[Dict[str, Any]]:
    """Assemble the public SearchResult from internal data."""
    pareto_results = compute_pareto_front(
        [result_to_candidate(r) for r in all_valid], objectives
    )
    pareto_configs = {prettify_configuration(c.configuration) for c in pareto_results}

    all_candidates: List[Candidate[Dict[str, Any]]] = []
    for cr, meta in all_candidates_meta:
        c_key = prettify_configuration(cr.configuration)
        meta_copy = dict(meta)
        if c_key in pareto_configs:
            meta_copy["is_pareto"] = True
        all_candidates.append(result_to_candidate(cr, meta_copy))

    pareto_list = [
        result_to_candidate(candidate_to_result(c), {"is_pareto": True})
        for c in pareto_results
    ]

    best_candidate = select_minimax_rank(pareto_results, objectives)
    if best_candidate is None:
        best_candidate = Candidate(configuration={}, objectives={}, metadata={})

    return SearchResult(
        objectives=list(objectives),
        best=best_candidate,
        pareto_front=pareto_list,
        all_candidates=all_candidates,
        history=history,
        best_per_generation=list(best_per_generation or []),
    )
