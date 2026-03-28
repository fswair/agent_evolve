"""Tests for Pareto dominance, front computation, and best-candidate selection."""

from agent_evolve.problem import ObjectiveSpec
from agent_evolve.results import (
    Candidate,
    SearchResult,
    compute_pareto_front,
    dominates,
    select_best_candidate,
    select_minimax_rank,
    sort_by_minimax_rank,
)

MAX_MIN = [ObjectiveSpec("accuracy", "max"), ObjectiveSpec("cost", "min")]
ONLY_MIN = [ObjectiveSpec("latency", "min"), ObjectiveSpec("cost", "min")]


# ------------------------------------------------------------------
# dominates
# ------------------------------------------------------------------

class TestDominates:
    def test_strictly_better(self):
        a = {"accuracy": 0.9, "cost": 1.0}
        b = {"accuracy": 0.8, "cost": 2.0}
        assert dominates(a, b, MAX_MIN)
        assert not dominates(b, a, MAX_MIN)

    def test_equal_not_dominating(self):
        a = {"accuracy": 0.9, "cost": 1.0}
        assert not dominates(a, a, MAX_MIN)

    def test_trade_off_no_dominance(self):
        a = {"accuracy": 0.9, "cost": 5.0}
        b = {"accuracy": 0.7, "cost": 1.0}
        assert not dominates(a, b, MAX_MIN)
        assert not dominates(b, a, MAX_MIN)

    def test_all_min(self):
        a = {"latency": 1.0, "cost": 2.0}
        b = {"latency": 3.0, "cost": 4.0}
        assert dominates(a, b, ONLY_MIN)


# ------------------------------------------------------------------
# compute_pareto_front
# ------------------------------------------------------------------

class TestParetoFront:
    def test_single_candidate(self):
        c = Candidate({"x": 1}, {"accuracy": 0.9, "cost": 1.0})
        front = compute_pareto_front([c], MAX_MIN)
        assert front == [c]

    def test_dominated_filtered(self):
        c1 = Candidate({"x": 1}, {"accuracy": 0.9, "cost": 1.0})
        c2 = Candidate({"x": 2}, {"accuracy": 0.8, "cost": 2.0})
        front = compute_pareto_front([c1, c2], MAX_MIN)
        assert front == [c1]

    def test_trade_off_both_kept(self):
        c1 = Candidate({"x": 1}, {"accuracy": 0.95, "cost": 5.0})
        c2 = Candidate({"x": 2}, {"accuracy": 0.80, "cost": 1.0})
        front = compute_pareto_front([c1, c2], MAX_MIN)
        assert len(front) == 2

    def test_empty(self):
        assert compute_pareto_front([], MAX_MIN) == []

    def test_three_candidates(self):
        c1 = Candidate({}, {"accuracy": 0.9, "cost": 2.0})
        c2 = Candidate({}, {"accuracy": 0.8, "cost": 1.0})
        c3 = Candidate({}, {"accuracy": 0.7, "cost": 3.0})  # dominated by both
        front = compute_pareto_front([c1, c2, c3], MAX_MIN)
        assert c1 in front
        assert c2 in front
        assert c3 not in front


# ------------------------------------------------------------------
# select_best_candidate
# ------------------------------------------------------------------

class TestSelectBest:
    def test_single(self):
        c = Candidate({}, {"accuracy": 0.9, "cost": 1.0})
        assert select_best_candidate([c], MAX_MIN) is c

    def test_prefers_max_first(self):
        c1 = Candidate({}, {"accuracy": 0.95, "cost": 5.0})
        c2 = Candidate({}, {"accuracy": 0.80, "cost": 1.0})
        best = select_best_candidate([c1, c2], MAX_MIN)
        assert best is c1

    def test_custom_priority(self):
        c1 = Candidate({}, {"accuracy": 0.95, "cost": 5.0})
        c2 = Candidate({}, {"accuracy": 0.80, "cost": 1.0})
        best = select_best_candidate([c1, c2], MAX_MIN, priority_order=["cost", "accuracy"])
        assert best is c2

    def test_empty(self):
        assert select_best_candidate([], MAX_MIN) is None


# ------------------------------------------------------------------
# select_minimax_rank
# ------------------------------------------------------------------

class TestMinimaxRank:
    def test_single(self):
        c = Candidate({}, {"accuracy": 0.9, "cost": 1.0})
        assert select_minimax_rank([c], MAX_MIN) is c

    def test_balanced_preferred(self):
        c_balanced = Candidate({}, {"accuracy": 0.85, "cost": 2.0})
        c_extreme1 = Candidate({}, {"accuracy": 0.99, "cost": 9.0})
        c_extreme2 = Candidate({}, {"accuracy": 0.50, "cost": 0.5})
        best = select_minimax_rank([c_balanced, c_extreme1, c_extreme2], MAX_MIN)
        assert best is c_balanced

    def test_dense_ranks_after_tie(self):
        """Two tied for best on an objective share rank 1; next distinct gets rank 2."""
        c1 = Candidate({"a": 1}, {"accuracy": 0.9, "cost": 1.0})
        c2 = Candidate({"a": 2}, {"accuracy": 0.9, "cost": 2.0})
        c3 = Candidate({"a": 3}, {"accuracy": 0.5, "cost": 3.0})
        best = select_minimax_rank([c1, c2, c3], MAX_MIN)
        assert best is c1

    def test_tie_break_by_sum_of_ranks(self):
        """Same minimax (worst) rank; prefer smaller sum of per-objective ranks."""
        c_lo = Candidate({}, {"accuracy": 0.82, "cost": 1.5})
        c_mid = Candidate({}, {"accuracy": 0.85, "cost": 3.0})
        c_hi = Candidate({}, {"accuracy": 0.86, "cost": 4.0})
        # accuracy (max): hi rank 1, mid rank 2, lo rank 3
        # cost (min): lo rank 1, mid rank 2, hi rank 3
        # worst: lo max(3,1)=3, mid max(2,2)=2, hi max(1,3)=3 → mid wins (only worst rank 2)
        assert select_minimax_rank([c_lo, c_mid, c_hi], MAX_MIN) is c_mid

        # Two with same worst rank 2, different sums: (2,1) sum 3 vs (2,2) sum 4
        c_a = Candidate({}, {"accuracy": 0.85, "cost": 3.0})
        c_b = Candidate({}, {"accuracy": 0.82, "cost": 1.5})
        c_c = Candidate({}, {"accuracy": 0.80, "cost": 2.0})
        # acc: a rank1, b rank2, c rank3
        # cost: b rank1, c rank2, a rank3
        # worst: a max(1,3)=3, b max(2,1)=2, c max(3,2)=3 → b alone
        assert select_minimax_rank([c_a, c_b, c_c], MAX_MIN) is c_b

        d1 = Candidate({}, {"accuracy": 0.84, "cost": 2.5})
        d2 = Candidate({}, {"accuracy": 0.84, "cost": 3.0})
        d3 = Candidate({}, {"accuracy": 0.83, "cost": 2.0})
        # acc: d1,d2 rank1 tie, d3 rank2
        # cost: d3 rank1, d1 rank2, d2 rank3
        # worst: d1 max(1,2)=2, d2 max(1,3)=3, d3 max(2,1)=2 → d1 and d3 tie (sum both 3)
        r = select_minimax_rank([d1, d2, d3], MAX_MIN)
        assert r is d1

    def test_empty(self):
        assert select_minimax_rank([], MAX_MIN) is None

    def test_sort_by_minimax_rank_first_matches_select(self):
        c_lo = Candidate({}, {"accuracy": 0.82, "cost": 1.5})
        c_mid = Candidate({}, {"accuracy": 0.85, "cost": 3.0})
        c_hi = Candidate({}, {"accuracy": 0.86, "cost": 4.0})
        xs = [c_lo, c_mid, c_hi]
        ordered = sort_by_minimax_rank(xs, MAX_MIN)
        assert ordered[0] is select_minimax_rank(xs, MAX_MIN)


# ------------------------------------------------------------------
# SearchResult
# ------------------------------------------------------------------

class TestSearchResult:
    def test_construction(self):
        c = Candidate({}, {"accuracy": 0.9, "cost": 1.0})
        sr = SearchResult(objectives=MAX_MIN, best=c, pareto_front=[c])
        assert sr.best is c
        assert len(sr.pareto_front) == 1
