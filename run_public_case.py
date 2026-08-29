"""Solve the calibrated U.S. linehaul application and its cost sensitivity.

Usage: python run_public_case.py
"""

from __future__ import annotations

import csv
from pathlib import Path

from aggregate_transport import (
    lift_solution,
    load_instance,
    solve_aggregate,
    with_class_cost_multiplier,
    write_aggregate_solution,
    write_lifted_solution,
)
from objective_functions import public_case_objective


ROOT = Path(__file__).resolve().parent


def main() -> None:
    instance = load_instance(ROOT)
    solution = solve_aggregate(instance)
    objective_recomputed = public_case_objective(
        instance, solution.flows, solution.counts
    )
    if abs(objective_recomputed - solution.objective) > 1e-6:
        raise ValueError("Independent objective recomputation failed")
    write_aggregate_solution(ROOT / "solution_aggregate.csv", instance, solution)
    write_lifted_solution(ROOT / "solution_lifted.csv", lift_solution(instance, solution))

    rows = []
    for multiplier in (0.70, 0.85, 1.00, 1.15, 1.30, 1.50):
        modified = with_class_cost_multiplier(instance, 1, multiplier)
        result = solve_aggregate(modified)
        used = result.counts.sum(axis=(0, 1))
        rows.append(
            {
                "class_2_cost_multiplier": multiplier,
                "objective": f"{result.objective:.6f}",
                "class_1_vehicles": int(used[0]),
                "class_2_vehicles": int(used[1]),
                "solver_status": result.status,
                "mip_gap": result.mip_gap,
            }
        )
    with (ROOT / "sensitivity_results_generated.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    used = solution.counts.sum(axis=(0, 1))
    print(f"objective={solution.objective:.6f}")
    print(f"objective_recomputed={objective_recomputed:.6f}")
    print(f"vehicles_used={used.tolist()}")
    print("wrote solution_aggregate.csv, solution_lifted.csv, and sensitivity_results_generated.csv")


if __name__ == "__main__":
    main()
