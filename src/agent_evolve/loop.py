"""Pareto-guided evolutionary optimisation loop.

Pure Python — receives LLM procedure callables as arguments,
knows nothing about kedi or adapters.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Protocol, Sequence

from agent_evolve.problem import ObjectiveSpec
from agent_evolve.results import (
    Candidate,
    SearchResult,
    compute_pareto_front,
    select_minimax_rank,
)
from agent_evolve._support import (
    CandidateResult,
    build_search_result,
    candidate_to_result,
    compute_performance_stats,
    evaluate_batch,
    parse_candidates,
    prettify_configuration,
    prettify_results,
    result_to_candidate,
    sample_failed_for_constraint,
)


class Procedures(Protocol):
    """Expected callable signatures for LLM procedures."""

    def generate_initial_candidates(
        self, search_space_desc: str, n_candidates: int,
    ) -> list: ...
    def regenerate_candidates(
        self, failed_str: str, n_candidates: int,
        search_space_desc: str, constraint_instruction: str,
        performance_insights: str,
    ) -> list: ...
    def generate_offspring(
        self, pareto_str: str, n_candidates: int,
        search_space_desc: str, constraint_instruction: str,
        performance_insights: str,
    ) -> list: ...
    def regenerate_offspring(
        self, failed_str: str, pareto_str: str, n_candidates: int,
        search_space_desc: str, constraint_instruction: str,
        performance_insights: str,
    ) -> list: ...
    def generate_failure_insights(
        self, failed_str: str, search_space_desc: str, n_failed: int,
    ) -> list: ...
    def generate_constraint_instruction(
        self, failed_str: str, search_space_desc: str,
    ) -> str: ...
    def update_constraint_instruction(
        self, previous_instruction: str, failed_str: str,
        search_space_desc: str,
    ) -> str: ...
    def generate_performance_insights(
        self, stats_str: str, search_space_desc: str,
    ) -> str: ...
    def update_performance_insights(
        self, previous_insights: str, pareto_str: str,
        total_valid: int, pareto_size: int,
    ) -> str: ...


LogFn = Callable[[str], None]

_DEFAULT_LLM_RETRIES = 3


def _noop_log(msg: str) -> None:
    pass


def _snapshot_best(
    pareto: Sequence[Candidate],
    objectives: Sequence[ObjectiveSpec],
) -> Candidate:
    """Minimax pick on the current Pareto set (same rule as ``SearchResult.best``)."""
    b = select_minimax_rank(list(pareto), objectives)
    if b is None:
        return Candidate(configuration={}, objectives={}, metadata={})
    return b


def _log_batch_metrics(
    log: LogFn,
    label: str,
    ordered: Sequence[CandidateResult],
    objectives: Sequence[ObjectiveSpec],
) -> None:
    """One line per candidate–metric pair; invalid candidates get a single error line."""
    if not ordered:
        log(f"{label}: (none)")
        return
    log(f"{label}:")
    for i, r in enumerate(ordered, 1):
        if r.is_valid:
            for spec in objectives:
                v = r.objectives.get(spec.name)
                log(f"  [{i}] {spec.name}: {v!s}")
        else:
            err = (r.error_message or "?").replace("\n", " ")
            if len(err) > 400:
                err = err[:400] + "..."
            log(f"  [{i}] error: {err}")


def _log_instruction_update(
    log: LogFn,
    title: str,
    previous: str,
    new: str,
) -> None:
    if new == previous:
        return
    log(f"{title}:")
    body = new.strip() if new.strip() else "(cleared)"
    for line in (body.splitlines() or [body]):
        log(f"  {line}")


def _retry(fn: Callable, args: tuple, retries: int, log: LogFn) -> Any:
    """Call *fn* with *args*, retrying up to *retries* times on failure."""
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return fn(*args)
        except Exception as exc:
            last_err = exc
            if attempt < retries:
                log(f"  [retry {attempt}/{retries}] {type(exc).__name__}: {exc}")
    raise last_err  # type: ignore[misc]


def run_evolution_loop(
    *,
    problem: Any,
    objectives: Sequence[ObjectiveSpec],
    search_space_desc: str,
    pop_size: int,
    generations: int,
    candidates_per_batch: int,
    max_regen_rounds: int,
    max_failed_examples: int,
    generate_initial_candidates: Callable,
    regenerate_candidates: Callable,
    generate_offspring: Callable,
    regenerate_offspring: Callable,
    generate_failure_insights: Callable,
    generate_constraint_instruction: Callable,
    update_constraint_instruction: Callable,
    generate_performance_insights: Callable,
    update_performance_insights: Callable,
    log: LogFn = _noop_log,
    llm_retries: int = _DEFAULT_LLM_RETRIES,
) -> SearchResult:
    """Run the full Pareto-guided evolutionary loop.

    All LLM interaction happens through the procedure callables.
    This function is pure Python orchestration.
    """
    if llm_retries > 1:
        def _w(fn: Callable) -> Callable:
            def wrapper(*a: Any) -> Any:
                return _retry(fn, a, llm_retries, log)
            return wrapper
        generate_initial_candidates = _w(generate_initial_candidates)
        regenerate_candidates = _w(regenerate_candidates)
        generate_offspring = _w(generate_offspring)
        regenerate_offspring = _w(regenerate_offspring)
        generate_failure_insights = _w(generate_failure_insights)
        generate_constraint_instruction = _w(generate_constraint_instruction)
        update_constraint_instruction = _w(update_constraint_instruction)
        generate_performance_insights = _w(generate_performance_insights)
        update_performance_insights = _w(update_performance_insights)

    all_valid: List[CandidateResult] = []
    all_failed: List[CandidateResult] = []
    all_candidates_meta: List[tuple] = []
    history: List[Dict[str, Any]] = []
    constraint_instruction = ""
    performance_insights = ""
    best_per_generation: List[Candidate] = []

    log(
        f"agent_evolve: {generations} generations, pop={pop_size}, "
        f"batch={candidates_per_batch}"
    )

    # -- Generation 1: initial sampling with regeneration ---------------
    gen1_valid, gen1_failed, constraint_instruction = _run_initial_generation(
        problem=problem,
        objectives=objectives,
        search_space_desc=search_space_desc,
        pop_size=pop_size,
        candidates_per_batch=candidates_per_batch,
        max_regen_rounds=max_regen_rounds,
        max_failed_examples=max_failed_examples,
        constraint_instruction=constraint_instruction,
        performance_insights=performance_insights,
        generate_initial_candidates=generate_initial_candidates,
        regenerate_candidates=regenerate_candidates,
        generate_failure_insights=generate_failure_insights,
        generate_constraint_instruction=generate_constraint_instruction,
        log=log,
    )

    all_valid.extend(gen1_valid)
    all_failed.extend(gen1_failed)
    for r in gen1_valid:
        all_candidates_meta.append((r, {"generation": 1, "is_pareto": False}))
    for r in gen1_failed:
        all_candidates_meta.append(
            (r, {"generation": 1, "is_pareto": False, "valid": False})
        )

    pareto = compute_pareto_front(
        [result_to_candidate(r) for r in all_valid], objectives,
    )
    best_per_generation.append(_snapshot_best(pareto, objectives))

    if gen1_valid:
        prev_pi = performance_insights
        performance_insights = _build_performance_insights(
            gen1_valid, objectives, search_space_desc,
            generate_performance_insights,
        )
        _log_instruction_update(
            log, "Performance instructions", prev_pi, performance_insights,
        )

    history.append({
        "gen": 1,
        "valid_count": len(gen1_valid),
        "failed_count": len(gen1_failed),
        "pareto_size": len(pareto),
    })

    # -- Generations 2..N: evolution from Pareto front -------------------
    for gen in range(2, generations + 1):
        gen_valid, gen_failed, constraint_instruction = _run_evolution_generation(
            problem=problem,
            objectives=objectives,
            search_space_desc=search_space_desc,
            pop_size=pop_size,
            candidates_per_batch=candidates_per_batch,
            max_regen_rounds=max_regen_rounds,
            max_failed_examples=max_failed_examples,
            prev_pareto=pareto,
            constraint_instruction=constraint_instruction,
            performance_insights=performance_insights,
            generate_initial_candidates=generate_initial_candidates,
            regenerate_candidates=regenerate_candidates,
            generate_offspring=generate_offspring,
            regenerate_offspring=regenerate_offspring,
            generate_failure_insights=generate_failure_insights,
            update_constraint_instruction=update_constraint_instruction,
            log=log,
        )

        all_valid.extend(gen_valid)
        all_failed.extend(gen_failed)
        for r in gen_valid:
            all_candidates_meta.append(
                (r, {"generation": gen, "is_pareto": False})
            )
        for r in gen_failed:
            all_candidates_meta.append(
                (r, {"generation": gen, "is_pareto": False, "valid": False})
            )

        pareto = compute_pareto_front(
            [result_to_candidate(r) for r in all_valid], objectives,
        )
        best_per_generation.append(_snapshot_best(pareto, objectives))

        if all_valid:
            stats = compute_performance_stats(all_valid, objectives)
            if stats:
                top_pareto = stats.get("top_3_pareto", [])
                p_str = (prettify_results(top_pareto, objectives)
                         if top_pareto else "None")
                prev_pi = performance_insights
                performance_insights = update_performance_insights(
                    performance_insights, p_str,
                    len(all_valid), stats.get("pareto_size", 0),
                )
                _log_instruction_update(
                    log, "Performance instructions", prev_pi, performance_insights,
                )

        history.append({
            "gen": gen,
            "valid_count": len(gen_valid),
            "failed_count": len(gen_failed),
            "pareto_size": len(pareto),
        })

    result = build_search_result(
        all_valid,
        all_candidates_meta,
        objectives,
        history,
        best_per_generation=best_per_generation,
    )
    log("")
    log(
        f"Summary: valid={len(all_valid)}, pareto={len(result.pareto_front)}, "
        f"final best objectives={result.best.objectives}"
    )
    log("Best per generation (minimax on cumulative Pareto):")
    for i, b in enumerate(result.best_per_generation, 1):
        cfg = prettify_configuration(b.configuration)
        log(f"  gen {i}: {b.objectives} | {cfg}")
    return result


# ------------------------------------------------------------------
# Generation helpers
# ------------------------------------------------------------------

def _run_initial_generation(
    *,
    problem: Any,
    objectives: Sequence[ObjectiveSpec],
    search_space_desc: str,
    pop_size: int,
    candidates_per_batch: int,
    max_regen_rounds: int,
    max_failed_examples: int,
    constraint_instruction: str,
    performance_insights: str,
    generate_initial_candidates: Callable,
    regenerate_candidates: Callable,
    generate_failure_insights: Callable,
    generate_constraint_instruction: Callable,
    log: LogFn,
) -> tuple:
    """Returns (valid, failed, constraint_instruction)."""
    gen_valid: List[CandidateResult] = []
    gen_failed: List[CandidateResult] = []
    last_round_failed: List[CandidateResult] = []

    for regen_round in range(max_regen_rounds):
        if regen_round == 0:
            raw = generate_initial_candidates(
                search_space_desc, candidates_per_batch,
            )
            batch_label = f"Proposed candidates (initial sampling, round {regen_round + 1})"
        else:
            failed_str = prettify_results(last_round_failed, objectives)
            ci = constraint_instruction or "No specific constraints learned yet."
            pi = performance_insights or "No performance insights available yet."
            raw = regenerate_candidates(
                failed_str, candidates_per_batch,
                search_space_desc, ci, pi,
            )
            batch_label = (
                f"Proposed candidates (regeneration, round {regen_round + 1})"
            )

        candidates, raw_elems = parse_candidates(raw, candidates_per_batch, log)
        valid_batch, failed_batch, ordered_batch = evaluate_batch(
            problem,
            candidates,
            objectives,
            raw_llm_elements=raw_elems,
            verbose=False,
            log_fn=log,
        )
        _log_batch_metrics(log, batch_label, ordered_batch, objectives)

        if failed_batch:
            _analyze_failures(
                failed_batch, objectives, search_space_desc,
                generate_failure_insights, log,
            )
            gen_failed.extend(failed_batch)

        gen_valid.extend(valid_batch)
        last_round_failed = failed_batch

        if gen_failed and (not constraint_instruction or valid_batch):
            sampled = sample_failed_for_constraint(
                last_round_failed, gen_failed, max_failed_examples,
            )
            sampled_str = prettify_results(sampled, objectives)
            prev_ci = constraint_instruction
            constraint_instruction = generate_constraint_instruction(
                sampled_str, search_space_desc,
            )
            _log_instruction_update(
                log, "Constraint instruction", prev_ci, constraint_instruction,
            )

        if len(gen_valid) >= pop_size:
            break

    gen_valid = gen_valid[:pop_size]
    return gen_valid, gen_failed, constraint_instruction


def _run_evolution_generation(
    *,
    problem: Any,
    objectives: Sequence[ObjectiveSpec],
    search_space_desc: str,
    pop_size: int,
    candidates_per_batch: int,
    max_regen_rounds: int,
    max_failed_examples: int,
    prev_pareto: Sequence[Candidate],
    constraint_instruction: str,
    performance_insights: str,
    generate_initial_candidates: Callable,
    regenerate_candidates: Callable,
    generate_offspring: Callable,
    regenerate_offspring: Callable,
    generate_failure_insights: Callable,
    update_constraint_instruction: Callable,
    log: LogFn,
) -> tuple:
    """Returns (valid, failed, constraint_instruction)."""
    gen_valid: List[CandidateResult] = []
    gen_failed: List[CandidateResult] = []

    if not prev_pareto:
        raw = generate_initial_candidates(search_space_desc, pop_size)
        first_label = "Proposed candidates (offspring, initial fallback batch)"
    else:
        pareto_as_cr = [
            CandidateResult(
                configuration=c.configuration,
                objectives=c.objectives, is_valid=True,
            )
            for c in prev_pareto
        ]
        pareto_str = prettify_results(pareto_as_cr[:5], objectives)
        ci = constraint_instruction or "Follow standard constraints."
        pi = (performance_insights
              or "Analyze the Pareto configurations for patterns.")
        raw = generate_offspring(
            pareto_str, pop_size, search_space_desc, ci, pi,
        )
        first_label = "Proposed candidates (offspring from Pareto)"

    candidates, raw_elems = parse_candidates(raw, pop_size, log)
    valid_batch, failed_batch, ordered_batch = evaluate_batch(
        problem,
        candidates,
        objectives,
        raw_llm_elements=raw_elems,
        verbose=False,
        log_fn=log,
    )
    _log_batch_metrics(log, first_label, ordered_batch, objectives)

    if failed_batch:
        _analyze_failures(
            failed_batch, objectives, search_space_desc,
            generate_failure_insights, log,
        )
        gen_failed.extend(failed_batch)
    gen_valid.extend(valid_batch)
    last_round_failed = failed_batch

    regen_round = 0
    while len(gen_valid) < pop_size and regen_round < max_regen_rounds:
        if not last_round_failed:
            break
        failed_str = prettify_results(last_round_failed, objectives)

        if prev_pareto:
            pareto_as_cr = [
                CandidateResult(
                    configuration=c.configuration,
                    objectives=c.objectives, is_valid=True,
                )
                for c in prev_pareto
            ]
            p_str = prettify_results(pareto_as_cr[:3], objectives)
            ci = (constraint_instruction
                  or "Follow the patterns from Pareto configurations.")
            pi = performance_insights or ""
            raw = regenerate_offspring(
                failed_str, p_str, candidates_per_batch,
                search_space_desc, ci, pi,
            )
            regen_label = (
                f"Proposed candidates (offspring regeneration {regen_round + 1})"
            )
        else:
            ci = constraint_instruction or "No specific constraints learned yet."
            pi = performance_insights or "No performance insights available yet."
            raw = regenerate_candidates(
                failed_str, candidates_per_batch,
                search_space_desc, ci, pi,
            )
            regen_label = (
                f"Proposed candidates (regeneration {regen_round + 1})"
            )

        candidates, raw_elems = parse_candidates(raw, candidates_per_batch, log)
        valid_batch, failed_batch, ordered_batch = evaluate_batch(
            problem,
            candidates,
            objectives,
            raw_llm_elements=raw_elems,
            verbose=False,
            log_fn=log,
        )
        _log_batch_metrics(log, regen_label, ordered_batch, objectives)

        if failed_batch:
            _analyze_failures(
                failed_batch, objectives, search_space_desc,
                generate_failure_insights, log,
            )
            gen_failed.extend(failed_batch)
            sampled = sample_failed_for_constraint(
                failed_batch, gen_failed, max_failed_examples,
            )
            sampled_str = prettify_results(sampled, objectives)
            prev_ci = constraint_instruction
            constraint_instruction = update_constraint_instruction(
                constraint_instruction, sampled_str, search_space_desc,
            )
            _log_instruction_update(
                log, "Constraint instruction", prev_ci, constraint_instruction,
            )

        gen_valid.extend(valid_batch)
        last_round_failed = failed_batch
        regen_round += 1

    gen_valid = gen_valid[:pop_size]
    return gen_valid, gen_failed, constraint_instruction


# ------------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------------

def _analyze_failures(
    failed: List[CandidateResult],
    objectives: Sequence[ObjectiveSpec],
    search_space_desc: str,
    generate_failure_insights: Callable,
    log: LogFn,
) -> None:
    failed_str = prettify_results(failed, objectives)
    insights = generate_failure_insights(
        failed_str, search_space_desc, len(failed),
    )
    if isinstance(insights, list):
        for r, insight in zip(failed, insights):
            r.insight = str(insight)
        if len(insights) != len(failed):
            log(
                f"[agent_evolve] WARNING: got {len(insights)} insights for "
                f"{len(failed)} failures — pairing may be wrong"
            )
    else:
        log(
            f"[agent_evolve] WARNING: generate_failure_insights returned "
            f"{type(insights).__name__}, expected list — insights not attached"
        )


def _build_performance_insights(
    valid_results: List[CandidateResult],
    objectives: Sequence[ObjectiveSpec],
    search_space_desc: str,
    generate_performance_insights: Callable,
) -> str:
    stats = compute_performance_stats(valid_results, objectives)
    if not stats:
        return ""
    lines: List[str] = []
    for spec in objectives:
        best = stats.get(f"best_{spec.name}")
        worst = stats.get(f"worst_{spec.name}")
        if best:
            lines.append(
                f"Best {spec.name}: {best.objectives.get(spec.name, 'N/A')}"
            )
            lines.append(
                f"  Config: {prettify_configuration(best.configuration)}"
            )
        if worst:
            lines.append(
                f"Worst {spec.name}: {worst.objectives.get(spec.name, 'N/A')}"
            )
    top_pareto = stats.get("top_3_pareto", [])
    if top_pareto:
        lines.append("\nTop Pareto configurations:")
        lines.append(prettify_results(top_pareto, objectives))
    stats_str = "\n".join(lines)
    return generate_performance_insights(stats_str, search_space_desc)
