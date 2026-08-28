"""Independent aggregate-versus-labeled validation on a compact test case.

Usage: python validate_equivalence.py
"""

from __future__ import annotations

from aggregate_transport import Destination, Instance, Origin, VehicleClass, lift_solution, solve_aggregate, solve_labeled


def main() -> None:
    toy = Instance(
        origins=(
            Origin("O1", 41.88, -87.63, 55.0),
            Origin("O2", 39.10, -94.58, 45.0),
        ),
        destinations=(
            Destination("D1", 40.71, -74.01, 40.0),
            Destination("D2", 33.45, -112.07, 35.0),
            Destination("D3", 29.76, -95.37, 25.0),
        ),
        vehicle_classes=(
            VehicleClass("C1", 20.0, 3, 0.075, 65.0, 0.28),
            VehicleClass("C2", 28.0, 2, 0.058, 85.0, 0.34),
        ),
    )
    aggregate = solve_aggregate(toy)
    labeled_objective = solve_labeled(toy)
    difference = abs(aggregate.objective - labeled_objective)
    lifted = lift_solution(toy, aggregate)
    lifted_flow = sum(float(row["load_tonnes"]) for row in lifted)
    expected_flow = sum(destination.demand_tonnes for destination in toy.destinations)

    assert difference <= 1e-6, (aggregate.objective, labeled_objective)
    assert abs(lifted_flow - expected_flow) <= 1e-6
    assert all(float(row["load_tonnes"]) >= -1e-9 for row in lifted)
    print(f"aggregate_objective={aggregate.objective:.9f}")
    print(f"labeled_objective={labeled_objective:.9f}")
    print(f"absolute_difference={difference:.3e}")
    print(f"lifted_vehicles={len(lifted)}")
    print("validation=PASS")


if __name__ == "__main__":
    main()
