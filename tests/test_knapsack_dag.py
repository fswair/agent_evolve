"""Tests for knapsack_dag: static table checks and ``evaluate_batch`` (validator + evaluator)."""

from __future__ import annotations

import sys
import unittest
from collections import deque
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "examples" / "knapsack_dag"))

from agent_evolve._support import evaluate_batch
from problem_def import DagSynergyKnapsackProblem


def _assert_requires_is_dag(cls: type[DagSynergyKnapsackProblem]) -> None:
    n = len(cls.ITEMS)
    for r, d in cls.REQUIRES:
        assert 0 <= r < n and 0 <= d < n, f"REQUIRES ({r}, {d}) out of range for n={n}"
        assert r != d, f"REQUIRES self-loop ({r}, {d})"

    graph: list[list[int]] = [[] for _ in range(n)]
    indegree = [0] * n
    for r, d in cls.REQUIRES:
        graph[r].append(d)
        indegree[d] += 1

    q = deque(i for i in range(n) if indegree[i] == 0)
    seen = 0
    while q:
        u = q.popleft()
        seen += 1
        for v in graph[u]:
            indegree[v] -= 1
            if indegree[v] == 0:
                q.append(v)
    assert seen == n, "REQUIRES induces a cycle (not a DAG)"


def _assert_synergy_table(cls: type[DagSynergyKnapsackProblem]) -> None:
    n = len(cls.ITEMS)
    seen_pairs: set[tuple[int, int]] = set()
    for t in cls.SYNERGY:
        assert len(t) == 3, "Each SYNERGY entry must be (i, j, bonus)"
        i, j, _bonus = t
        if i > j:
            i, j = j, i
        assert 0 <= i < n and 0 <= j < n, f"SYNERGY indices ({i}, {j}) out of range for n={n}"
        assert i != j, f"SYNERGY self-pair ({i}, {j})"
        pair = (i, j)
        assert pair not in seen_pairs, f"Duplicate SYNERGY pair after normalization: {pair}"
        seen_pairs.add(pair)


class TestStaticTables(unittest.TestCase):
    def test_items_nonempty(self):
        self.assertGreater(len(DagSynergyKnapsackProblem.ITEMS), 0)

    def test_requires_dag(self):
        _assert_requires_is_dag(DagSynergyKnapsackProblem)

    def test_synergy(self):
        _assert_synergy_table(DagSynergyKnapsackProblem)


class TestGoldenEvaluate(unittest.TestCase):
    def test_reference_configurations(self):
        p = DagSynergyKnapsackProblem()
        p.validate({"selection": []})
        self.assertEqual(
            p.evaluate({"selection": []}),
            {
                "total_score": 0.0,
                "total_weight": 0.0,
                "synergy_count": 0.0,
            },
        )
        p.validate({"selection": [0, 1, 2, 5]})
        self.assertEqual(
            p.evaluate({"selection": [0, 1, 2, 5]}),
            {
                "total_score": 240.0,
                "total_weight": 50.0,
                "synergy_count": 1.0,
            },
        )


class TestEvaluateBatch(unittest.TestCase):
    """Same path as the evolver: ``validate`` then ``evaluate``."""

    def setUp(self):
        self.problem = DagSynergyKnapsackProblem()
        self.objectives = list(self.problem.objectives)

    def test_valid_mixed_batch(self):
        candidates = [
            {"selection": []},
            {"selection": [0, 1, 2, 5]},
            {"selection": [3, 4, 6, 7, 8]},
        ]
        valid, failed, ordered = evaluate_batch(
            self.problem, candidates, self.objectives, verbose=False
        )
        self.assertEqual(len(ordered), 3)
        self.assertEqual(len(failed), 0)
        self.assertEqual(len(valid), 3)
        self.assertEqual(
            valid[0].objectives,
            {
                "total_score": 0.0,
                "total_weight": 0.0,
                "synergy_count": 0.0,
            },
        )
        self.assertEqual(valid[1].objectives["total_score"], 240.0)
        self.assertEqual(valid[2].objectives["total_score"], 298.0)
        self.assertEqual(valid[2].objectives["synergy_count"], 3.0)

    def test_dependency_violation_fails(self):
        valid, failed, _ = evaluate_batch(
            self.problem,
            [{"selection": [2]}],
            self.objectives,
            verbose=False,
        )
        self.assertEqual(len(valid), 0)
        self.assertEqual(len(failed), 1)
        self.assertFalse(failed[0].is_valid)
        self.assertIn("requires item 0", failed[0].error_message or "")

    def test_capacity_violation_fails(self):
        valid, failed, _ = evaluate_batch(
            self.problem,
            [{"selection": [0, 1, 2, 5, 7]}],
            self.objectives,
            verbose=False,
        )
        self.assertEqual(len(valid), 0)
        self.assertEqual(len(failed), 1)
        self.assertIn("exceeds capacity", failed[0].error_message or "")

    def test_duplicate_index_fails(self):
        valid, failed, _ = evaluate_batch(
            self.problem,
            [{"selection": [0, 0]}],
            self.objectives,
            verbose=False,
        )
        self.assertEqual(len(valid), 0)
        self.assertEqual(len(failed), 1)
        self.assertIn("Duplicate", failed[0].error_message or "")


if __name__ == "__main__":
    unittest.main()
