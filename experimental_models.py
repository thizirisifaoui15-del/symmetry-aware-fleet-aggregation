"""Reproducible multi-commodity aggregate and vehicle-labeled MILP models.

The module implements exactly the formulations stated in the paper.  It is
used by the experiment drivers and deliberately keeps model construction
independent from table and figure generation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter
from typing import Iterable

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


@dataclass(frozen=True)
class ExperimentalInstance:
    name: str
    seed: int
    origin_coordinates: np.ndarray
    destination_coordinates: np.ndarray
    supply: np.ndarray
    demand: np.ndarray
    capacities: np.ndarray
    fleet_sizes: np.ndarray
    flow_objectives: np.ndarray
    dispatch_objectives: np.ndarray

    @property
    def dimensions(self) -> tuple[int, int, int, int, int]:
        p = int(self.flow_objectives.shape[0])
        i, j, k, g = map(int, self.flow_objectives.shape[1:])
        return p, i, j, k, g

    def with_fleet_sizes(self, fleet_sizes: Iterable[int], name: str | None = None):
        fleet_sizes = np.asarray(tuple(fleet_sizes), dtype=int)
        if fleet_sizes.shape != self.fleet_sizes.shape or np.any(fleet_sizes < 0):
            raise ValueError("fleet_sizes has an invalid shape or value")
        return replace(
            self,
            name=self.name if name is None else name,
            fleet_sizes=fleet_sizes,
        )


@dataclass
class ModelSolution:
    formulation: str
    objective_index: int
    objective: float | None
    objective_vector: np.ndarray | None
    status: int
    message: str
    optimal: bool
    has_incumbent: bool
    mip_gap: float | None
    node_count: int | None
    build_seconds: float
    solve_seconds: float
    total_seconds: float
    variable_count: int
    integer_variable_count: int
    constraint_count: int
    flows: np.ndarray | None
    dispatch: np.ndarray | None


def euclidean_distances(instance: ExperimentalInstance) -> np.ndarray:
    delta = (
        instance.origin_coordinates[:, None, :]
        - instance.destination_coordinates[None, :, :]
    )
    return np.sqrt(np.sum(delta * delta, axis=2))


def evaluate_aggregate(
    instance: ExperimentalInstance, flows: np.ndarray, dispatch: np.ndarray
) -> np.ndarray:
    return np.sum(
        instance.flow_objectives * flows[None, ...], axis=(1, 2, 3, 4)
    ) + np.sum(
        instance.dispatch_objectives * dispatch[None, ...], axis=(1, 2, 3)
    )


def evaluate_labeled(
    instance: ExperimentalInstance, flows: np.ndarray, dispatch: np.ndarray
) -> np.ndarray:
    class_of_vehicle = np.repeat(
        np.arange(len(instance.fleet_sizes)), instance.fleet_sizes
    )
    flow_coefficients = instance.flow_objectives[..., class_of_vehicle]
    dispatch_coefficients = instance.dispatch_objectives[..., class_of_vehicle]
    return np.sum(
        flow_coefficients * flows[None, ...], axis=(1, 2, 3, 4)
    ) + np.sum(
        dispatch_coefficients * dispatch[None, ...], axis=(1, 2, 3)
    )


def formulation_size(
    instance: ExperimentalInstance,
    formulation: str,
    symmetry_breaking: bool = False,
) -> tuple[int, int, int]:
    _, ni, nj, nk, ng = instance.dimensions
    if formulation == "aggregate":
        variables = ni * nj * ng * (nk + 1)
        integers = ni * nj * ng
        constraints = ni * nk + nj * nk + ni * nj * ng + ng
        return variables, integers, constraints
    if formulation != "labeled":
        raise ValueError("formulation must be 'aggregate' or 'labeled'")
    vehicles = int(np.sum(instance.fleet_sizes))
    variables = ni * nj * vehicles * (nk + 1)
    integers = ni * nj * vehicles
    constraints = ni * nk + nj * nk + ni * nj * vehicles + vehicles
    if symmetry_breaking:
        constraints += int(np.sum(np.maximum(instance.fleet_sizes - 1, 0)))
    return variables, integers, constraints


def _matrix(
    rows: list[int],
    cols: list[int],
    values: list[float],
    row_count: int,
    column_count: int,
):
    return coo_matrix(
        (values, (rows, cols)), shape=(row_count, column_count)
    ).tocsr()


def _finite_float(value: object) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def solve_experimental_model(
    instance: ExperimentalInstance,
    formulation: str,
    objective_index: int = 0,
    time_limit: float = 20.0,
    relax: bool = False,
    epsilon_bounds: dict[int, float] | None = None,
    symmetry_breaking: bool = False,
    mip_rel_gap: float = 1e-7,
    presolve: bool = True,
) -> ModelSolution:
    """Build and solve one aggregate or labeled scalarization."""

    p, ni, nj, nk, ng = instance.dimensions
    if not 0 <= objective_index < p:
        raise ValueError("objective_index is out of range")
    if formulation not in {"aggregate", "labeled"}:
        raise ValueError("unknown formulation")
    epsilon_bounds = {} if epsilon_bounds is None else dict(epsilon_bounds)
    if objective_index in epsilon_bounds:
        raise ValueError("the primary objective cannot also be epsilon-bounded")

    build_start = perf_counter()
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    row = 0

    if formulation == "aggregate":
        flow_count = ni * nj * nk * ng
        dispatch_count = ni * nj * ng
        variable_count = flow_count + dispatch_count

        def xidx(i: int, j: int, k: int, g: int) -> int:
            return ((i * nj + j) * nk + k) * ng + g

        def yidx(i: int, j: int, g: int) -> int:
            return flow_count + (i * nj + j) * ng + g

        objective_vectors = np.concatenate(
            [
                instance.flow_objectives.reshape(p, flow_count),
                instance.dispatch_objectives.reshape(p, dispatch_count),
            ],
            axis=1,
        )
        lower_bounds = np.zeros(variable_count)
        upper_bounds = np.full(variable_count, np.inf)
        for i in range(ni):
            for j in range(nj):
                for g in range(ng):
                    upper_bounds[yidx(i, j, g)] = instance.fleet_sizes[g]
        integrality = np.zeros(variable_count, dtype=int)
        if not relax:
            integrality[flow_count:] = 1

        for i in range(ni):
            for k in range(nk):
                for j in range(nj):
                    for g in range(ng):
                        rows.append(row); cols.append(xidx(i, j, k, g)); values.append(1.0)
                lower.append(-np.inf); upper.append(float(instance.supply[i, k])); row += 1
        for j in range(nj):
            for k in range(nk):
                for i in range(ni):
                    for g in range(ng):
                        rows.append(row); cols.append(xidx(i, j, k, g)); values.append(1.0)
                demand = float(instance.demand[j, k])
                lower.append(demand); upper.append(demand); row += 1
        for i in range(ni):
            for j in range(nj):
                for g in range(ng):
                    for k in range(nk):
                        rows.append(row); cols.append(xidx(i, j, k, g)); values.append(1.0)
                    rows.append(row); cols.append(yidx(i, j, g)); values.append(-float(instance.capacities[g]))
                    lower.append(-np.inf); upper.append(0.0); row += 1
        for g in range(ng):
            for i in range(ni):
                for j in range(nj):
                    rows.append(row); cols.append(yidx(i, j, g)); values.append(1.0)
            lower.append(-np.inf); upper.append(float(instance.fleet_sizes[g])); row += 1

        output_shape_flow = (ni, nj, nk, ng)
        output_shape_dispatch = (ni, nj, ng)
        integer_count = 0 if relax else dispatch_count

    else:
        class_of_vehicle = np.repeat(np.arange(ng), instance.fleet_sizes)
        vehicle_count = len(class_of_vehicle)
        flow_count = ni * nj * nk * vehicle_count
        dispatch_count = ni * nj * vehicle_count
        variable_count = flow_count + dispatch_count

        def xidx(i: int, j: int, k: int, v: int) -> int:
            return ((i * nj + j) * nk + k) * vehicle_count + v

        def yidx(i: int, j: int, v: int) -> int:
            return flow_count + (i * nj + j) * vehicle_count + v

        objective_vectors = np.concatenate(
            [
                instance.flow_objectives[..., class_of_vehicle].reshape(p, flow_count),
                instance.dispatch_objectives[..., class_of_vehicle].reshape(
                    p, dispatch_count
                ),
            ],
            axis=1,
        )
        lower_bounds = np.zeros(variable_count)
        upper_bounds = np.concatenate(
            [np.full(flow_count, np.inf), np.ones(dispatch_count)]
        )
        integrality = np.zeros(variable_count, dtype=int)
        if not relax:
            integrality[flow_count:] = 1

        for i in range(ni):
            for k in range(nk):
                for j in range(nj):
                    for v in range(vehicle_count):
                        rows.append(row); cols.append(xidx(i, j, k, v)); values.append(1.0)
                lower.append(-np.inf); upper.append(float(instance.supply[i, k])); row += 1
        for j in range(nj):
            for k in range(nk):
                for i in range(ni):
                    for v in range(vehicle_count):
                        rows.append(row); cols.append(xidx(i, j, k, v)); values.append(1.0)
                demand = float(instance.demand[j, k])
                lower.append(demand); upper.append(demand); row += 1
        for i in range(ni):
            for j in range(nj):
                for v, g in enumerate(class_of_vehicle):
                    for k in range(nk):
                        rows.append(row); cols.append(xidx(i, j, k, v)); values.append(1.0)
                    rows.append(row); cols.append(yidx(i, j, v)); values.append(-float(instance.capacities[g]))
                    lower.append(-np.inf); upper.append(0.0); row += 1
        for v in range(vehicle_count):
            for i in range(ni):
                for j in range(nj):
                    rows.append(row); cols.append(yidx(i, j, v)); values.append(1.0)
            lower.append(-np.inf); upper.append(1.0); row += 1
        if symmetry_breaking:
            first_vehicle = 0
            for class_size in instance.fleet_sizes:
                for local in range(int(class_size) - 1):
                    left = first_vehicle + local
                    right = left + 1
                    for i in range(ni):
                        for j in range(nj):
                            rows.extend([row, row])
                            cols.extend([yidx(i, j, right), yidx(i, j, left)])
                            values.extend([1.0, -1.0])
                    lower.append(-np.inf); upper.append(0.0); row += 1
                first_vehicle += int(class_size)

        output_shape_flow = (ni, nj, nk, vehicle_count)
        output_shape_dispatch = (ni, nj, vehicle_count)
        integer_count = 0 if relax else dispatch_count

    for criterion, bound in sorted(epsilon_bounds.items()):
        if not 0 <= criterion < p:
            raise ValueError("epsilon criterion is out of range")
        coefficient = objective_vectors[criterion]
        nonzero = np.flatnonzero(coefficient)
        rows.extend([row] * len(nonzero))
        cols.extend(nonzero.tolist())
        values.extend(coefficient[nonzero].tolist())
        lower.append(-np.inf); upper.append(float(bound)); row += 1

    matrix = _matrix(rows, cols, values, row, variable_count)
    build_seconds = perf_counter() - build_start
    solve_start = perf_counter()
    result = milp(
        c=objective_vectors[objective_index],
        integrality=integrality,
        bounds=Bounds(lower_bounds, upper_bounds),
        constraints=LinearConstraint(
            matrix, np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
        ),
        options={
            "time_limit": float(time_limit),
            "mip_rel_gap": float(mip_rel_gap),
            "presolve": bool(presolve),
        },
    )
    solve_seconds = perf_counter() - solve_start
    has_incumbent = result.x is not None and np.all(np.isfinite(result.x))
    flows = dispatch = objective_vector = None
    objective = None
    if has_incumbent:
        flows = np.asarray(result.x[:flow_count]).reshape(output_shape_flow)
        dispatch = np.asarray(result.x[flow_count:]).reshape(output_shape_dispatch)
        if not relax:
            dispatch = np.rint(dispatch)
        if formulation == "aggregate":
            objective_vector = evaluate_aggregate(instance, flows, dispatch)
        else:
            objective_vector = evaluate_labeled(instance, flows, dispatch)
        objective = float(objective_vector[objective_index])

    return ModelSolution(
        formulation=("SB" if symmetry_breaking else ("A" if formulation == "aggregate" else "L")),
        objective_index=objective_index,
        objective=objective,
        objective_vector=objective_vector,
        status=int(result.status),
        message=str(result.message),
        optimal=int(result.status) == 0,
        has_incumbent=bool(has_incumbent),
        mip_gap=_finite_float(getattr(result, "mip_gap", None)),
        node_count=(
            None
            if getattr(result, "mip_node_count", None) is None
            else int(result.mip_node_count)
        ),
        build_seconds=float(build_seconds),
        solve_seconds=float(solve_seconds),
        total_seconds=float(build_seconds + solve_seconds),
        variable_count=int(variable_count),
        integer_variable_count=int(integer_count),
        constraint_count=int(row),
        flows=flows,
        dispatch=dispatch,
    )


def lift_aggregate_solution(
    instance: ExperimentalInstance, solution: ModelSolution
) -> tuple[list[dict[str, object]], np.ndarray]:
    """Construct vehicle-level loading records and independently value them."""

    if solution.formulation != "A" or solution.flows is None or solution.dispatch is None:
        raise ValueError("an aggregate incumbent is required")
    p, ni, nj, nk, ng = instance.dimensions
    records: list[dict[str, object]] = []
    vector = np.zeros(p)
    for g in range(ng):
        next_vehicle = 0
        for i in range(ni):
            for j in range(nj):
                count = int(round(float(solution.dispatch[i, j, g])))
                if count <= 0:
                    continue
                selected = list(range(next_vehicle, next_vehicle + count))
                next_vehicle += count
                if next_vehicle > int(instance.fleet_sizes[g]):
                    raise ValueError("lift exceeds the available fleet")
                remaining_capacity = np.full(count, float(instance.capacities[g]))
                vehicle_position = 0
                for k in range(nk):
                    amount = float(solution.flows[i, j, k, g])
                    while amount > 1e-8:
                        while (
                            vehicle_position < count
                            and remaining_capacity[vehicle_position] <= 1e-8
                        ):
                            vehicle_position += 1
                        if vehicle_position >= count:
                            raise ValueError("lifted flow exceeds selected capacity")
                        load = min(amount, remaining_capacity[vehicle_position])
                        records.append(
                            {
                                "class": g,
                                "vehicle": selected[vehicle_position],
                                "origin": i,
                                "destination": j,
                                "commodity": k,
                                "load": load,
                            }
                        )
                        vector += instance.flow_objectives[:, i, j, k, g] * load
                        remaining_capacity[vehicle_position] -= load
                        amount -= load
                vector += instance.dispatch_objectives[:, i, j, g] * count
    return records, vector

