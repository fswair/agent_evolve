"""Dependency knapsack with pairwise synergies — problem definition.

Items form a DAG of requirements: if item ``j`` is carried, every item ``i``
such that ``(i, j)`` is in ``REQUIRES`` must also be carried. Some item pairs
grant an extra score when both are present.

Satisfies the agent_evolve contract: ``problem``, ``config``, ``CandidateConfig``.
"""

from pydantic import BaseModel, Field

from agent_evolve import ObjectiveSpec


class CandidateConfig(BaseModel):
    """One candidate configuration; fields must match ``validate`` / ``evaluate``."""

    selection: list[int] = Field(
        default_factory=list,
        description="Item indices (empty list allowed)",
    )


class DagSynergyKnapsackProblem:
    candidate_model = CandidateConfig
    """0/1 knapsack with dependency edges and additive pairwise bonuses."""

    # (weight, base_value)
    ITEMS = [
        (8, 40),
        (10, 35),
        (12, 50),
        (6, 25),
        (15, 60),
        (20, 90),
        (5, 30),
        (18, 70),
        (9, 45),
        (11, 55),
    ]
    CAPACITY = 58

    # If ``dependent`` is selected, ``required`` must be selected (DAG edges).
    REQUIRES = [
        (0, 2),
        (1, 2),
        (2, 5),
        (3, 4),
        (6, 8),
    ]

    # (i, j, bonus) with i < j; bonus applies iff both indices are selected.
    SYNERGY = [
        (0, 2, 25),
        (1, 3, 15),
        (3, 4, 30),
        (6, 7, 20),
        (4, 7, 18),
        (2, 6, 12),
    ]

    @property
    def objectives(self):
        return [
            ObjectiveSpec("total_score", "max"),
            ObjectiveSpec("total_weight", "min"),
            ObjectiveSpec("synergy_count", "max"),
        ]

    def _normalized_synergy(self):
        out = []
        for t in self.SYNERGY:
            if len(t) != 3:
                raise ValueError("Each SYNERGY entry must be (i, j, bonus)")
            i, j, bonus = t
            if i > j:
                i, j = j, i
            out.append((i, j, float(bonus)))
        return out

    def validate(self, config):
        selection = config.get("selection", [])
        if not isinstance(selection, list):
            raise ValueError("'selection' must be a list of item indices")
        if not all(isinstance(i, int) for i in selection):
            raise ValueError("Each selection entry must be an int")
        if len(selection) != len(set(selection)):
            raise ValueError("Duplicate item indices are not allowed")
        n = len(self.ITEMS)
        for i in selection:
            if i < 0 or i >= n:
                raise ValueError(f"Invalid item index {i}. Must be in 0..{n - 1}")
        sel_set = set(selection)
        for required, dependent in self.REQUIRES:
            if dependent in sel_set and required not in sel_set:
                raise ValueError(
                    f"Dependency: item {dependent} requires item {required} to be selected"
                )
        weight = sum(self.ITEMS[i][0] for i in selection)
        if weight > self.CAPACITY:
            raise ValueError(
                f"Total weight {weight} exceeds capacity {self.CAPACITY}"
            )
        return True

    def evaluate(self, config):
        selection = config["selection"]
        sel_set = set(selection)
        base = sum(self.ITEMS[i][1] for i in selection)
        syn_bonus = 0.0
        syn_count = 0
        for i, j, bonus in self._normalized_synergy():
            if i in sel_set and j in sel_set:
                syn_bonus += bonus
                syn_count += 1
        weight = sum(self.ITEMS[i][0] for i in selection)
        return {
            "total_score": float(base + syn_bonus),
            "total_weight": float(weight),
            "synergy_count": float(syn_count),
        }

    def search_space_description(self):
        lines = [
            "Dependency knapsack with synergies",
            "",
            "Select a subset of items by index (0/1). Each index at most once.",
            'Configuration: {"selection": [0, 2, 5, ...]}',
            "Empty selection is allowed (zero score and weight).",
            "",
            "Rules:",
            "  - REQUIRES: for each (required, dependent), if dependent is in the",
            "    selection then required must also be in the selection.",
            "  - Total weight must not exceed CAPACITY.",
            "",
            f"CAPACITY = {self.CAPACITY}",
            "",
            "Items (index: weight, base_value):",
        ]
        for idx, (w, v) in enumerate(self.ITEMS):
            lines.append(f"  {idx}: weight={w}, base_value={v}")
        lines.extend(
            [
                "",
                "REQUIRES (required, dependent):",
            ]
        )
        for a, b in self.REQUIRES:
            lines.append(f"  ({a}, {b})  — if {b} is taken, {a} must be taken")
        lines.extend(
            [
                "",
                "SYNERGY (i, j, bonus): if both i and j are taken, add bonus to total_score.",
                "synergy_count is the number of such pairs satisfied.",
            ]
        )
        for i, j, b in self._normalized_synergy():
            lines.append(f"  ({i}, {j}): +{b}")
        lines.extend(
            [
                "",
                "Objectives:",
                "  total_score = sum(base_value) + sum(synergy bonuses for satisfied pairs)",
                "  total_weight = sum(weight)",
                "  synergy_count = number of synergy pairs with both ends selected",
            ]
        )
        return "\n".join(lines)


problem = DagSynergyKnapsackProblem()

config = {
    "pop_size": 16,
    "generations": 4,
    "candidates_per_batch": 4,
    "max_regen_rounds": 5,
    "max_failed_examples": 3,
    "verbose": False,
}
