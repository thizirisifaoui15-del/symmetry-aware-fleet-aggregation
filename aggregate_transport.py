"""Exact aggregate and labeled MILP models for homogeneous single-trip fleets.

The data format uses one divisible commodity (tonnes).  The aggregate model has
one continuous flow and one bounded integer vehicle count for every
origin-destination-class tuple.  The labeled model is included as an independent
small-instance validation reference.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


@dataclass(frozen=True)
class Origin:
    name: str
    latitude: float
    longitude: float
    stock_tonnes: float


@dataclass(frozen=True)
class Destination:
    name: str
    latitude: float
    longitude: float
    demand_tonnes: float


@dataclass(frozen=True)
class VehicleClass:
    name: str
    capacity_tonnes: float
    vehicles: int
    variable_cost_per_tonne_mile: float
    dispatch_base_cost: float
    dispatch_cost_per_mile: float


@dataclass(frozen=True)
class Instance:
    origins: tuple[Origin, ...]
    destinations: tuple[Destination, ...]
    vehicle_classes: tuple[VehicleClass, ...]
    road_distance_multiplier: float = 1.18


@dataclass
class AggregateSolution:
    objective: float
    status: int
    message: str
    mip_gap: float | None
    flows: np.ndarray
    counts: np.ndarray
    distances_miles: np.ndarray


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_instance(data_directory: str | Path) -> Instance:
    data_directory = Path(data_directory)
    origins = tuple(
        Origin(
            row["origin"],
            float(row["latitude"]),
            float(row["longitude"]),
            float(row["stock_tonnes"]),
        )
        for row in _read_csv(data_directory / "public_us_origins.csv")
    )
    destinations = tuple(
        Destination(
            row["destination"],
            float(row["latitude"]),
            float(row["longitude"]),
            float(row["demand_tonnes"]),
        )
        for row in _read_csv(data_directory / "public_us_destinations.csv")
    )
    classes = tuple(
        VehicleClass(
            row["vehicle_class"],
            float(row["capacity_tonnes"]),
            int(row["vehicles"]),
            float(row["variable_cost_per_tonne_mile"]),
            float(row["dispatch_base_cost"]),
            float(row["dispatch_cost_per_mile"]),
        )
        for row in _read_csv(data_directory / "vehicle_classes.csv")
    )
    return Instance(origins, destinations, classes)


def with_class_cost_multiplier(instance: Instance, class_index: int, multiplier: float) -> Instance:
    classes = list(instance.vehicle_classes)
    classes[class_index] = replace(
        classes[class_index],
        variable_cost_per_tonne_mile=(
            classes[class_index].variable_cost_per_tonne_mile * multiplier
        ),
    )
    return replace(instance, vehicle_classes=tuple(classes))


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_miles = 3958.7613
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * radius_miles * math.asin(math.sqrt(a))


def distance_matrix(instance: Instance) -> np.ndarray:
    return np.array(
        [
            [
                instance.road_distance_multiplier
                * haversine_miles(o.latitude, o.longitude, d.latitude, d.longitude)
                for d in instance.destinations
            ]
            for o in instance.origins
        ],
        dtype=float,
    )


def _constraint_matrix(rows: list[int], cols: list[int], values: list[float], row_count: int, col_count: int):
    return coo_matrix((values, (rows, cols)), shape=(row_count, col_count)).tocsr()


def solve_aggregate(instance: Instance, time_limit: float = 20.0) -> AggregateSolution:
    origins, destinations, classes = instance.origins, instance.destinations, instance.vehicle_classes
    ni, nj, ng = len(origins), len(destinations), len(classes)
    arc_count = ni * nj * ng
    variable_count = 2 * arc_count
    distances = distance_matrix(instance)

    def arc(i: int, j: int, g: int) -> int:
        return (i * nj + j) * ng + g

    objective = np.zeros(variable_count)
    for i in range(ni):
        for j in range(nj):
            for g, vehicle in enumerate(classes):
                a = arc(i, j, g)
                objective[a] = vehicle.variable_cost_per_tonne_mile * distances[i, j]
                objective[arc_count + a] = vehicle.dispatch_base_cost + vehicle.dispatch_cost_per_mile * distances[i, j]

    lower_bounds = np.zeros(variable_count)
    upper_bounds = np.full(variable_count, np.inf)
    for i in range(ni):
        for j in range(nj):
            for g, vehicle in enumerate(classes):
                upper_bounds[arc_count + arc(i, j, g)] = vehicle.vehicles
    integrality = np.concatenate([np.zeros(arc_count, dtype=int), np.ones(arc_count, dtype=int)])

    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    constraint_lower: list[float] = []
    constraint_upper: list[float] = []
    row = 0

    for i, origin in enumerate(origins):
        for j in range(nj):
            for g in range(ng):
                rows.append(row); cols.append(arc(i, j, g)); values.append(1.0)
        constraint_lower.append(-np.inf); constraint_upper.append(origin.stock_tonnes); row += 1

    for j, destination in enumerate(destinations):
        for i in range(ni):
            for g in range(ng):
                rows.append(row); cols.append(arc(i, j, g)); values.append(1.0)
        constraint_lower.append(destination.demand_tonnes); constraint_upper.append(destination.demand_tonnes); row += 1

    for i in range(ni):
        for j in range(nj):
            for g, vehicle in enumerate(classes):
                a = arc(i, j, g)
                rows.extend([row, row]); cols.extend([a, arc_count + a]); values.extend([1.0, -vehicle.capacity_tonnes])
                constraint_lower.append(-np.inf); constraint_upper.append(0.0); row += 1

    for g, vehicle in enumerate(classes):
        for i in range(ni):
            for j in range(nj):
                rows.append(row); cols.append(arc_count + arc(i, j, g)); values.append(1.0)
        constraint_lower.append(-np.inf); constraint_upper.append(float(vehicle.vehicles)); row += 1

    matrix = _constraint_matrix(rows, cols, values, row, variable_count)
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower_bounds, upper_bounds),
        constraints=LinearConstraint(matrix, np.array(constraint_lower), np.array(constraint_upper)),
        options={"time_limit": time_limit, "mip_rel_gap": 1e-9, "presolve": True},
    )
    if result.x is None:
        raise RuntimeError(f"Aggregate MILP returned no feasible solution: {result.message}")
    return AggregateSolution(
        objective=float(result.fun),
        status=int(result.status),
        message=str(result.message),
        mip_gap=None if getattr(result, "mip_gap", None) is None else float(result.mip_gap),
        flows=np.asarray(result.x[:arc_count]).reshape(ni, nj, ng),
        counts=np.rint(result.x[arc_count:]).astype(int).reshape(ni, nj, ng),
        distances_miles=distances,
    )


def solve_labeled(instance: Instance, time_limit: float = 20.0) -> float:
    origins, destinations, classes = instance.origins, instance.destinations, instance.vehicle_classes
    ni, nj = len(origins), len(destinations)
    vehicles = [(g, r) for g, vehicle in enumerate(classes) for r in range(vehicle.vehicles)]
    nv = len(vehicles)
    assignment_count = ni * nj * nv
    variable_count = 2 * assignment_count
    distances = distance_matrix(instance)

    def aidx(i: int, j: int, v: int) -> int:
        return (i * nj + j) * nv + v

    objective = np.zeros(variable_count)
    for i in range(ni):
        for j in range(nj):
            for v, (g, _) in enumerate(vehicles):
                vehicle = classes[g]
                a = aidx(i, j, v)
                objective[a] = vehicle.variable_cost_per_tonne_mile * distances[i, j]
                objective[assignment_count + a] = vehicle.dispatch_base_cost + vehicle.dispatch_cost_per_mile * distances[i, j]

    bounds = Bounds(np.zeros(variable_count), np.concatenate([np.full(assignment_count, np.inf), np.ones(assignment_count)]))
    integrality = np.concatenate([np.zeros(assignment_count, dtype=int), np.ones(assignment_count, dtype=int)])
    rows: list[int] = []; cols: list[int] = []; values: list[float] = []
    lower: list[float] = []; upper: list[float] = []; row = 0

    for i, origin in enumerate(origins):
        for j in range(nj):
            for v in range(nv):
                rows.append(row); cols.append(aidx(i, j, v)); values.append(1.0)
        lower.append(-np.inf); upper.append(origin.stock_tonnes); row += 1
    for j, destination in enumerate(destinations):
        for i in range(ni):
            for v in range(nv):
                rows.append(row); cols.append(aidx(i, j, v)); values.append(1.0)
        lower.append(destination.demand_tonnes); upper.append(destination.demand_tonnes); row += 1
    for i in range(ni):
        for j in range(nj):
            for v, (g, _) in enumerate(vehicles):
                a = aidx(i, j, v)
                rows.extend([row, row]); cols.extend([a, assignment_count + a]); values.extend([1.0, -classes[g].capacity_tonnes])
                lower.append(-np.inf); upper.append(0.0); row += 1
    for v in range(nv):
        for i in range(ni):
            for j in range(nj):
                rows.append(row); cols.append(assignment_count + aidx(i, j, v)); values.append(1.0)
        lower.append(-np.inf); upper.append(1.0); row += 1

    matrix = _constraint_matrix(rows, cols, values, row, variable_count)
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=bounds,
        constraints=LinearConstraint(matrix, np.array(lower), np.array(upper)),
        options={"time_limit": time_limit, "mip_rel_gap": 1e-9, "presolve": True},
    )
    if result.x is None:
        raise RuntimeError(f"Labeled MILP returned no feasible solution: {result.message}")
    return float(result.fun)


def lift_solution(instance: Instance, solution: AggregateSolution) -> list[dict[str, object]]:
    assignments: list[dict[str, object]] = []
    for g, vehicle in enumerate(instance.vehicle_classes):
        next_vehicle = 1
        for i, origin in enumerate(instance.origins):
            for j, destination in enumerate(instance.destinations):
                count = int(solution.counts[i, j, g])
                remaining = float(solution.flows[i, j, g])
                for _ in range(count):
                    load = min(vehicle.capacity_tonnes, max(0.0, remaining))
                    assignments.append(
                        {
                            "vehicle_class": vehicle.name,
                            "vehicle_id": f"{vehicle.name}-{next_vehicle:02d}",
                            "origin": origin.name,
                            "destination": destination.name,
                            "load_tonnes": load,
                        }
                    )
                    next_vehicle += 1
                    remaining -= load
                if remaining > 1e-6:
                    raise ValueError("Aggregate flow could not be lifted within selected vehicle capacity")
        if next_vehicle - 1 > vehicle.vehicles:
            raise ValueError("Lift uses more vehicles than the fleet contains")
    return assignments


def write_aggregate_solution(path: str | Path, instance: Instance, solution: AggregateSolution) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["origin", "destination", "vehicle_class", "distance_miles", "vehicles", "flow_tonnes"])
        for i, origin in enumerate(instance.origins):
            for j, destination in enumerate(instance.destinations):
                for g, vehicle in enumerate(instance.vehicle_classes):
                    count = int(solution.counts[i, j, g])
                    flow = float(solution.flows[i, j, g])
                    if count or flow > 1e-8:
                        writer.writerow([origin.name, destination.name, vehicle.name, f"{solution.distances_miles[i,j]:.3f}", count, f"{flow:.6f}"])


def write_lifted_solution(path: str | Path, assignments: Iterable[dict[str, object]]) -> None:
    assignments = list(assignments)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["vehicle_class", "vehicle_id", "origin", "destination", "load_tonnes"])
        writer.writeheader()
        writer.writerows(assignments)
