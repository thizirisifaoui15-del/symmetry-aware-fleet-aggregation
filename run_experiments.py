"""Run every computational experiment reported in the paper.

Examples
--------
Run the complete campaign::

    python run_experiments.py --experiments all

Run selected blocks::

    python run_experiments.py --experiments main clone public

Every solve produces a row-level CSV record and a JSONL solver log.  Summary
tables are generated only from those raw records.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
from dataclasses import asdict
from pathlib import Path
from statistics import median

import numpy as np
import scipy

from aggregate_transport import distance_matrix, load_instance
from experimental_models import (
    ExperimentalInstance,
    ModelSolution,
    formulation_size,
    lift_aggregate_solution,
    solve_experimental_model,
)


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RAW = RESULTS / "raw"
TABLES = RESULTS / "tables"
LOGS = RESULTS / "logs"

MAIN_FAMILIES = (
    ("S1", 3, 4, 3),
    ("S2", 3, 5, 4),
    ("S3", 4, 6, 5),
    ("S4", 5, 7, 6),
    ("S5", 6, 8, 8),
    ("S6", 7, 10, 10),
)
MAIN_SEEDS = (1101, 1102, 1103)
CLONE_SEED = 271828
ROBUSTNESS_SEEDS = tuple(range(2201, 2211))
LARGE_FLEET_SEEDS = (3301, 3302, 3303)
TRIOBJECTIVE_SEED = 314159


def ensure_directories() -> None:
    for directory in (RAW, TABLES, LOGS):
        directory.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path.name}")
    fieldnames = list(rows[0])
    for row in rows[1:]:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_jsonl(path: Path, record: dict[str, object]) -> None:
    def encode(value: object):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(f"cannot JSON-encode {type(value).__name__}")

    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, default=encode) + "\n")


def reset_log(name: str) -> Path:
    path = LOGS / f"{name}.jsonl"
    path.write_text("", encoding="utf-8")
    return path


def solution_record(
    experiment: str,
    instance: ExperimentalInstance,
    solution: ModelSolution,
    **extra: object,
) -> dict[str, object]:
    p, ni, nj, nk, ng = instance.dimensions
    vector = (
        None
        if solution.objective_vector is None
        else [float(x) for x in solution.objective_vector]
    )
    record: dict[str, object] = {
        "experiment": experiment,
        "instance": instance.name,
        "seed": instance.seed,
        "origins": ni,
        "destinations": nj,
        "commodities": nk,
        "classes": ng,
        "fleet_sizes": [int(x) for x in instance.fleet_sizes],
        "formulation": solution.formulation,
        "objective_index": solution.objective_index,
        "objective": solution.objective,
        "objective_vector": vector,
        "status": solution.status,
        "message": solution.message,
        "optimal": solution.optimal,
        "has_incumbent": solution.has_incumbent,
        "mip_gap": solution.mip_gap,
        "node_count": solution.node_count,
        "build_seconds": solution.build_seconds,
        "solve_seconds": solution.solve_seconds,
        "total_seconds": solution.total_seconds,
        "variables": solution.variable_count,
        "integer_variables": solution.integer_variable_count,
        "constraints": solution.constraint_count,
    }
    record.update(extra)
    return record


def csv_record(record: dict[str, object]) -> dict[str, object]:
    clean = dict(record)
    for key, value in clean.items():
        if isinstance(value, (list, tuple, dict)):
            clean[key] = json.dumps(value, separators=(",", ":"))
    return clean


def save_solution(
    log_path: Path,
    raw_rows: list[dict[str, object]],
    experiment: str,
    instance: ExperimentalInstance,
    solution: ModelSolution,
    **extra: object,
) -> None:
    record = solution_record(experiment, instance, solution, **extra)
    append_jsonl(log_path, record)
    raw_rows.append(csv_record(record))


def generate_instance(
    name: str,
    seed: int,
    origins: int,
    destinations: int,
    commodities: int,
    classes: int,
    fleet_sizes: tuple[int, ...],
    demand_anchor: tuple[int, ...] | None = None,
    demand_ratio: float = 0.58,
) -> ExperimentalInstance:
    rng = np.random.default_rng(seed)
    origin_coordinates = rng.uniform(0.0, 100.0, size=(origins, 2))
    destination_coordinates = rng.uniform(0.0, 100.0, size=(destinations, 2))
    distances = np.sqrt(
        np.sum(
            (origin_coordinates[:, None, :] - destination_coordinates[None, :, :])
            ** 2,
            axis=2,
        )
    ) + 10.0

    capacity_catalog = np.asarray([55.0, 85.0, 70.0, 100.0])
    variable_catalog = np.asarray([0.075, 0.058, 0.066, 0.051])
    base_catalog = np.asarray([65.0, 85.0, 75.0, 100.0])
    distance_dispatch_catalog = np.asarray([0.28, 0.34, 0.30, 0.38])
    if classes > len(capacity_catalog):
        raise ValueError("the controlled generator supports at most four classes")
    capacities = capacity_catalog[:classes]
    fleet = np.asarray(fleet_sizes, dtype=int)
    if len(fleet) != classes:
        raise ValueError("one fleet size is required per class")
    anchor = fleet if demand_anchor is None else np.asarray(demand_anchor, dtype=int)
    total_demand = demand_ratio * float(np.sum(anchor * capacities))
    commodity_share = rng.dirichlet(np.full(commodities, 2.0))
    demand = np.zeros((destinations, commodities))
    for k in range(commodities):
        destination_share = rng.dirichlet(np.full(destinations, 1.5))
        demand[:, k] = total_demand * commodity_share[k] * destination_share
    supply = heterogeneous_supply(rng, demand, origins)

    flow_objectives = np.zeros((1, origins, destinations, commodities, classes))
    dispatch_objectives = np.zeros((1, origins, destinations, classes))
    for k in range(commodities):
        for g in range(classes):
            flow_objectives[0, :, :, k, g] = (
                distances * variable_catalog[g] * (1.0 + 0.08 * k)
            )
    for g in range(classes):
        dispatch_objectives[0, :, :, g] = (
            base_catalog[g] + distance_dispatch_catalog[g] * distances
        )
    return ExperimentalInstance(
        name=name,
        seed=seed,
        origin_coordinates=origin_coordinates,
        destination_coordinates=destination_coordinates,
        supply=supply,
        demand=demand,
        capacities=capacities,
        fleet_sizes=fleet,
        flow_objectives=flow_objectives,
        dispatch_objectives=dispatch_objectives,
    )


def heterogeneous_supply(
    rng: np.random.Generator,
    demand: np.ndarray,
    origins: int,
    total_supply_factor: float = 1.10,
) -> np.ndarray:
    """Generate reproducible, nonredundant origin stocks for every commodity.

    Total system stock is 10% above demand.  A seeded pair of origins is used
    for all commodities: the dominant origin receives 85% of stock and the
    secondary origin 15%.  The dominant origin still holds only 93.5% of
    system demand, so both origins must ship.  Reusing one pair avoids making
    small instances infeasible merely by requiring too many distinct trips.
    """
    if origins < 2:
        raise ValueError("at least two origins are required")
    if total_supply_factor <= 1.0:
        raise ValueError("total_supply_factor must exceed one")
    commodity_totals = np.sum(demand, axis=0)
    supply = np.zeros((origins, demand.shape[1]))
    dominant, secondary = rng.choice(origins, size=2, replace=False)
    for commodity, total in enumerate(commodity_totals):
        shares = np.zeros(origins)
        shares[dominant] = 0.85
        shares[secondary] = 0.15
        supply[:, commodity] = total_supply_factor * total * shares
    if np.any(np.max(supply, axis=0) >= commodity_totals):
        raise AssertionError("a single origin can cover the full commodity demand")
    return supply


def generate_triobjective_instance() -> ExperimentalInstance:
    seed = TRIOBJECTIVE_SEED
    rng = np.random.default_rng(seed)
    origins, destinations, commodities, classes = 3, 5, 2, 3
    origin_coordinates = rng.uniform(0.0, 100.0, size=(origins, 2))
    destination_coordinates = rng.uniform(0.0, 100.0, size=(destinations, 2))
    distances = np.sqrt(
        np.sum(
            (origin_coordinates[:, None, :] - destination_coordinates[None, :, :])
            ** 2,
            axis=2,
        )
    ) + 10.0
    capacities = np.asarray([55.0, 72.0, 60.0])
    fleet = np.asarray([4, 4, 4])
    total_demand = 0.52 * float(np.sum(capacities * fleet))
    commodity_share = rng.dirichlet(np.asarray([2.0, 2.0]))
    demand = np.zeros((destinations, commodities))
    for k in range(commodities):
        demand[:, k] = (
            total_demand
            * commodity_share[k]
            * rng.dirichlet(np.full(destinations, 1.8))
        )
    supply = heterogeneous_supply(rng, demand, origins)

    flow_objectives = np.zeros((3, origins, destinations, commodities, classes))
    dispatch_objectives = np.zeros((3, origins, destinations, classes))
    economic_rate = np.asarray([0.052, 0.070, 0.087])
    environmental_rate = np.asarray([0.112, 0.048, 0.082])
    time_rate = np.asarray([0.020, 0.017, 0.010])
    economic_base = np.asarray([46.0, 62.0, 78.0])
    environmental_base = np.asarray([36.0, 18.0, 28.0])
    time_base = np.asarray([15.0, 13.0, 8.0])
    for k in range(commodities):
        commodity_factor = 1.0 + 0.10 * k
        for g in range(classes):
            flow_objectives[0, :, :, k, g] = distances * economic_rate[g] * commodity_factor
            flow_objectives[1, :, :, k, g] = distances * environmental_rate[g] * commodity_factor
            flow_objectives[2, :, :, k, g] = distances * time_rate[g] * commodity_factor
    for g in range(classes):
        dispatch_objectives[0, :, :, g] = economic_base[g] + 0.22 * distances
        dispatch_objectives[1, :, :, g] = environmental_base[g] + 0.08 * distances
        dispatch_objectives[2, :, :, g] = time_base[g] + 0.04 * distances
    return ExperimentalInstance(
        name="triobjective_seed_314159",
        seed=seed,
        origin_coordinates=origin_coordinates,
        destination_coordinates=destination_coordinates,
        supply=supply,
        demand=demand,
        capacities=capacities,
        fleet_sizes=fleet,
        flow_objectives=flow_objectives,
        dispatch_objectives=dispatch_objectives,
    )


def public_instance() -> ExperimentalInstance:
    source = load_instance(ROOT)
    distances = distance_matrix(source)
    ni, nj, ng = len(source.origins), len(source.destinations), len(source.vehicle_classes)
    flow = np.zeros((1, ni, nj, 1, ng))
    dispatch = np.zeros((1, ni, nj, ng))
    for g, vehicle in enumerate(source.vehicle_classes):
        flow[0, :, :, 0, g] = vehicle.variable_cost_per_tonne_mile * distances
        dispatch[0, :, :, g] = vehicle.dispatch_base_cost + vehicle.dispatch_cost_per_mile * distances
    return ExperimentalInstance(
        name="public_us_linehaul",
        seed=0,
        origin_coordinates=np.asarray(
            [(x.latitude, x.longitude) for x in source.origins], dtype=float
        ),
        destination_coordinates=np.asarray(
            [(x.latitude, x.longitude) for x in source.destinations], dtype=float
        ),
        supply=np.asarray([[x.stock_tonnes] for x in source.origins], dtype=float),
        demand=np.asarray([[x.demand_tonnes] for x in source.destinations], dtype=float),
        capacities=np.asarray([x.capacity_tonnes for x in source.vehicle_classes]),
        fleet_sizes=np.asarray([x.vehicles for x in source.vehicle_classes], dtype=int),
        flow_objectives=flow,
        dispatch_objectives=dispatch,
    )


def run_main() -> None:
    experiment = "main"
    log = reset_log(experiment)
    raw_rows: list[dict[str, object]] = []
    for family_index, (family, ni, nj, m) in enumerate(MAIN_FAMILIES):
        for seed in MAIN_SEEDS:
            actual_seed = seed + 100 * family_index
            instance = generate_instance(
                f"{family}_seed_{actual_seed}", actual_seed, ni, nj, 2, 2, (m, m)
            )
            for formulation in ("aggregate", "labeled"):
                solution = solve_experimental_model(
                    instance, formulation, time_limit=5.0, mip_rel_gap=1e-7
                )
                save_solution(log, raw_rows, experiment, instance, solution, family=family)
            print(f"main {family} seed {actual_seed}: complete", flush=True)
    write_csv(RAW / "main_runs.csv", raw_rows)

    summary: list[dict[str, object]] = []
    for family, ni, nj, m in MAIN_FAMILIES:
        rows = [r for r in raw_rows if r["family"] == family]
        aggregate = {int(r["seed"]): r for r in rows if r["formulation"] == "A"}
        labeled = {int(r["seed"]): r for r in rows if r["formulation"] == "L"}
        gaps = []
        for seed, row in labeled.items():
            if bool(row["has_incumbent"]):
                a = float(aggregate[seed]["objective"])
                gaps.append(max(0.0, 100.0 * (float(row["objective"]) - a) / a))
        t_a = median(float(r["total_seconds"]) for r in aggregate.values())
        t_l = median(float(r["total_seconds"]) for r in labeled.values())
        n_a = int(next(iter(aggregate.values()))["variables"])
        n_l = int(next(iter(labeled.values()))["variables"])
        summary.append(
            {
                "family": family,
                "origins": ni,
                "destinations": nj,
                "vehicles_per_class": m,
                "labeled_variables": n_l,
                "aggregate_variables": n_a,
                "reduction_percent": 100.0 * (1.0 - n_a / n_l),
                "labeled_time_seconds": t_l,
                "aggregate_time_seconds": t_a,
                "speedup": t_l / t_a,
                "aggregate_optima": f"{sum(bool(r['optimal']) for r in aggregate.values())}/3",
                "labeled_optima": f"{sum(bool(r['optimal']) for r in labeled.values())}/3",
                "max_incumbent_gap_percent": max(gaps) if gaps else "",
            }
        )
    write_csv(TABLES / "benchmark_main_generated.csv", summary)


def run_clone() -> None:
    experiment = "clone"
    log = reset_log(experiment)
    raw_rows: list[dict[str, object]] = []
    base = generate_instance(
        "clone_base", CLONE_SEED, 5, 7, 2, 2, (4, 4), demand_anchor=(4, 4), demand_ratio=0.40
    )
    for m in (4, 6, 8, 10, 12, 16, 20, 24, 30):
        instance = base.with_fleet_sizes((m, m), f"clone_m_{m}")
        for formulation in ("aggregate", "labeled"):
            solution = solve_experimental_model(instance, formulation, time_limit=3.0)
            save_solution(log, raw_rows, experiment, instance, solution, vehicles_per_class=m)
        print(f"clone m={m}: complete", flush=True)
    write_csv(RAW / "clone_runs.csv", raw_rows)
    summary: list[dict[str, object]] = []
    for m in (4, 6, 8, 10, 12, 16, 20, 24, 30):
        rows = [r for r in raw_rows if int(r["vehicles_per_class"]) == m]
        a = next(r for r in rows if r["formulation"] == "A")
        l = next(r for r in rows if r["formulation"] == "L")
        summary.append(
            {
                "vehicles_per_class": m,
                "log10_symmetry_group": 2.0 * math.lgamma(m + 1) / math.log(10.0),
                "labeled_variables": l["variables"],
                "aggregate_variables": a["variables"],
                "labeled_time_seconds": l["total_seconds"],
                "aggregate_time_seconds": a["total_seconds"],
                "aggregate_objective": a["objective"],
                "labeled_objective": l["objective"],
                "labeled_optimal": l["optimal"],
            }
        )
    write_csv(TABLES / "clone_sensitivity_generated.csv", summary)


def run_robustness() -> None:
    experiment = "robustness"
    log = reset_log(experiment)
    raw_rows: list[dict[str, object]] = []
    for m in (10, 20, 30):
        for seed in ROBUSTNESS_SEEDS:
            instance = generate_instance(
                f"robust_m_{m}_seed_{seed}",
                seed,
                5,
                7,
                2,
                2,
                (m, m),
                demand_anchor=(8, 8),
            )
            settings = (
                ("aggregate", False),
                ("labeled", False),
                ("labeled", True),
            )
            for formulation, sb in settings:
                solution = solve_experimental_model(
                    instance,
                    formulation,
                    symmetry_breaking=sb,
                    time_limit=30.0,
                    mip_rel_gap=1e-7,
                )
                save_solution(
                    log,
                    raw_rows,
                    experiment,
                    instance,
                    solution,
                    vehicles_per_class=m,
                )
            print(f"robustness m={m} seed={seed}: complete", flush=True)
    write_csv(RAW / "robustness_runs.csv", raw_rows)
    summary: list[dict[str, object]] = []
    for m in (10, 20, 30):
        for formulation in ("A", "L", "SB"):
            rows = [
                r
                for r in raw_rows
                if int(r["vehicles_per_class"]) == m
                and r["formulation"] == formulation
            ]
            times = np.asarray([float(r["solve_seconds"]) for r in rows])
            summary.append(
                {
                    "vehicles_per_class": m,
                    "formulation": formulation,
                    "variables": rows[0]["variables"],
                    "integer_variables": rows[0]["integer_variables"],
                    "median_seconds": float(np.median(times)),
                    "q1_seconds": float(np.quantile(times, 0.25)),
                    "q3_seconds": float(np.quantile(times, 0.75)),
                    "proven_optima": f"{sum(bool(r['optimal']) for r in rows)}/10",
                    "incumbents": f"{sum(bool(r['has_incumbent']) for r in rows)}/10",
                }
            )
    write_csv(TABLES / "robustness_generated.csv", summary)


def run_large_fleet() -> None:
    experiment = "large_fleet"
    log = reset_log(experiment)
    raw_rows: list[dict[str, object]] = []
    families = (("LF2", 2, 2), ("LF3", 3, 3))
    for family, commodities, classes in families:
        for seed in LARGE_FLEET_SEEDS:
            base = generate_instance(
                f"{family}_seed_{seed}_base",
                seed,
                5,
                7,
                commodities,
                classes,
                tuple([8] * classes),
                demand_anchor=tuple([8] * classes),
            )
            for m in (50, 100, 200, 500):
                instance = base.with_fleet_sizes(
                    tuple([m] * classes), f"{family}_m_{m}_seed_{seed}"
                )
                for formulation in ("aggregate", "labeled"):
                    milp_solution = solve_experimental_model(
                        instance, formulation, time_limit=15.0, mip_rel_gap=1e-7
                    )
                    save_solution(
                        log,
                        raw_rows,
                        experiment,
                        instance,
                        milp_solution,
                        family=family,
                        vehicles_per_class=m,
                        relaxation="MILP",
                    )
                    lp_solution = solve_experimental_model(
                        instance, formulation, time_limit=45.0, relax=True
                    )
                    save_solution(
                        log,
                        raw_rows,
                        experiment,
                        instance,
                        lp_solution,
                        family=family,
                        vehicles_per_class=m,
                        relaxation="LP",
                    )
                print(f"large fleet {family} m={m} seed={seed}: complete", flush=True)
    write_csv(RAW / "large_fleet_runs.csv", raw_rows)

    size_rows: list[dict[str, object]] = []
    performance_rows: list[dict[str, object]] = []
    for family, commodities, classes in families:
        exemplar = generate_instance(
            f"{family}_size", 1, 5, 7, commodities, classes, tuple([8] * classes)
        )
        for m in (50, 100, 200, 500):
            instance = exemplar.with_fleet_sizes(tuple([m] * classes))
            n_a, int_a, cons_a = formulation_size(instance, "aggregate")
            n_l, int_l, cons_l = formulation_size(instance, "labeled")
            size_rows.append(
                {
                    "family": family,
                    "commodities": commodities,
                    "classes": classes,
                    "vehicles_per_class": m,
                    "aggregate_variables": n_a,
                    "labeled_variables": n_l,
                    "aggregate_integer": int_a,
                    "labeled_integer": int_l,
                    "aggregate_constraints": cons_a,
                    "labeled_constraints": cons_l,
                    "reduction_percent": 100.0 * (1.0 - n_a / n_l),
                }
            )
            subset = [
                r
                for r in raw_rows
                if r["family"] == family and int(r["vehicles_per_class"]) == m
            ]
            a_mip = {int(r["seed"]): r for r in subset if r["formulation"] == "A" and r["relaxation"] == "MILP"}
            l_mip = {int(r["seed"]): r for r in subset if r["formulation"] == "L" and r["relaxation"] == "MILP"}
            a_lp = {int(r["seed"]): r for r in subset if r["formulation"] == "A" and r["relaxation"] == "LP"}
            l_lp = {int(r["seed"]): r for r in subset if r["formulation"] == "L" and r["relaxation"] == "LP"}
            lp_gaps = []
            incumbent_gaps = []
            lp_differences = []
            for seed in LARGE_FLEET_SEEDS:
                if a_mip[seed]["objective"] not in (None, "") and a_lp[seed]["objective"] not in (None, ""):
                    optimum = float(a_mip[seed]["objective"])
                    lp_gaps.append(100.0 * (optimum - float(a_lp[seed]["objective"])) / optimum)
                if l_mip[seed]["objective"] not in (None, ""):
                    optimum = float(a_mip[seed]["objective"])
                    incumbent_gaps.append(100.0 * (float(l_mip[seed]["objective"]) - optimum) / optimum)
                if a_lp[seed]["objective"] not in (None, "") and l_lp[seed]["objective"] not in (None, ""):
                    lp_differences.append(abs(float(a_lp[seed]["objective"]) - float(l_lp[seed]["objective"])))
            performance_rows.append(
                {
                    "family": family,
                    "vehicles_per_class": m,
                    "aggregate_time_seconds": median(float(r["total_seconds"]) for r in a_mip.values()),
                    "labeled_time_seconds": median(float(r["total_seconds"]) for r in l_mip.values()),
                    "aggregate_optima": sum(bool(r["optimal"]) for r in a_mip.values()),
                    "labeled_optima": sum(bool(r["optimal"]) for r in l_mip.values()),
                    "labeled_incumbents": f"{sum(bool(r['has_incumbent']) for r in l_mip.values())}/3",
                    "lp_gap_percent": median(lp_gaps),
                    "max_incumbent_gap_percent": max(incumbent_gaps) if incumbent_gaps else "",
                    "lp_comparisons": len(lp_differences),
                    "max_lp_difference": max(lp_differences) if lp_differences else "",
                    "labeled_lp_time_seconds": median(float(r["total_seconds"]) for r in l_lp.values()),
                }
            )
    write_csv(TABLES / "large_fleet_sizes_generated.csv", size_rows)
    write_csv(TABLES / "large_fleet_performance_generated.csv", performance_rows)


def nondominated(records: list[dict[str, object]], tolerance: float = 1e-7):
    kept: list[dict[str, object]] = []
    for candidate in records:
        vector = np.asarray(candidate["objective_vector"], dtype=float)
        dominated = False
        for other in records:
            other_vector = np.asarray(other["objective_vector"], dtype=float)
            if np.all(other_vector <= vector + tolerance) and np.any(other_vector < vector - tolerance):
                dominated = True
                break
        if not dominated:
            kept.append(candidate)
    return kept


def run_triobjective() -> None:
    experiment = "triobjective"
    log = reset_log(experiment)
    raw_rows: list[dict[str, object]] = []
    instance = generate_triobjective_instance()
    payoff_rows: list[dict[str, object]] = []
    payoff_solutions: list[ModelSolution] = []
    names = ("economic_cost", "environmental_impact", "transit_time_burden")
    for criterion, name in enumerate(names):
        solution = solve_experimental_model(
            instance, "aggregate", objective_index=criterion, time_limit=20.0
        )
        if not solution.optimal:
            raise RuntimeError(f"triobjective payoff solve {name} was not certified")
        save_solution(log, raw_rows, experiment, instance, solution, stage="payoff", criterion=name)
        payoff_solutions.append(solution)
        payoff_rows.append(
            {
                "single_criterion_solve": name,
                "economic_cost": solution.objective_vector[0],
                "environmental_impact": solution.objective_vector[1],
                "transit_time_burden": solution.objective_vector[2],
            }
        )
    write_csv(TABLES / "triobjective_payoff_generated.csv", payoff_rows)

    payoff_vectors = np.asarray([s.objective_vector for s in payoff_solutions])
    environmental_grid = np.linspace(
        float(np.min(payoff_vectors[:, 1])), float(np.max(payoff_vectors[:, 1])), 9
    )
    time_grid = np.linspace(
        float(np.min(payoff_vectors[:, 2])), float(np.max(payoff_vectors[:, 2])), 9
    )
    feasible: list[dict[str, object]] = []
    for env_index, env_bound in enumerate(environmental_grid):
        for time_index, time_bound in enumerate(time_grid):
            solution = solve_experimental_model(
                instance,
                "aggregate",
                objective_index=0,
                time_limit=20.0,
                epsilon_bounds={1: float(env_bound), 2: float(time_bound)},
            )
            save_solution(
                log,
                raw_rows,
                experiment,
                instance,
                solution,
                stage="epsilon_grid",
                environmental_bound=float(env_bound),
                time_bound=float(time_bound),
                environmental_index=env_index,
                time_index=time_index,
            )
            if solution.has_incumbent:
                feasible.append(
                    {
                        "environmental_bound": float(env_bound),
                        "time_bound": float(time_bound),
                        "objective_vector": [float(x) for x in solution.objective_vector],
                        "total_seconds": solution.total_seconds,
                        "solution": solution,
                    }
                )
        print(f"triobjective grid row {env_index + 1}/9: complete", flush=True)

    unique: dict[tuple[float, float, float], dict[str, object]] = {}
    for record in feasible:
        key = tuple(np.round(record["objective_vector"], 7))
        unique.setdefault(key, record)
    frontier = nondominated(list(unique.values()))
    frontier.sort(key=lambda r: tuple(r["objective_vector"]))
    frontier_rows: list[dict[str, object]] = []
    max_lift_difference = 0.0
    for point_index, record in enumerate(frontier):
        solution = record["solution"]
        _, lifted_vector = lift_aggregate_solution(instance, solution)
        difference = float(np.max(np.abs(lifted_vector - solution.objective_vector)))
        max_lift_difference = max(max_lift_difference, difference)
        frontier_rows.append(
            {
                "point": point_index,
                "economic_cost": solution.objective_vector[0],
                "environmental_impact": solution.objective_vector[1],
                "transit_time_burden": solution.objective_vector[2],
                "environmental_bound": record["environmental_bound"],
                "time_bound": record["time_bound"],
                "aggregate_total_seconds": record["total_seconds"],
                "lift_max_absolute_difference": difference,
            }
        )
    write_csv(TABLES / "triobjective_frontier_generated.csv", frontier_rows)

    if not frontier:
        raise RuntimeError("triobjective grid produced no nondominated point")
    check_indices = sorted(set(np.linspace(0, len(frontier) - 1, min(10, len(frontier)), dtype=int)))
    labeled_differences = []
    labeled_times = []
    for index in check_indices:
        record = frontier[index]
        solution = solve_experimental_model(
            instance,
            "labeled",
            objective_index=0,
            time_limit=20.0,
            epsilon_bounds={
                1: float(record["environmental_bound"]),
                2: float(record["time_bound"]),
            },
        )
        save_solution(
            log,
            raw_rows,
            experiment,
            instance,
            solution,
            stage="labeled_check",
            frontier_point=index,
            environmental_bound=record["environmental_bound"],
            time_bound=record["time_bound"],
        )
        if solution.has_incumbent:
            difference = abs(float(solution.objective) - float(record["objective_vector"][0]))
            labeled_differences.append(difference)
            labeled_times.append(solution.total_seconds)

    write_csv(RAW / "triobjective_runs.csv", raw_rows)
    summary = [
        {
            "seed": TRIOBJECTIVE_SEED,
            "grid_subproblems": 81,
            "feasible_grid_subproblems": len(feasible),
            "unique_vectors": len(unique),
            "sample_nondominated_vectors": len(frontier),
            "labeled_checks": len(check_indices),
            "max_lift_objective_difference": max_lift_difference,
            "max_labeled_primary_difference": max(labeled_differences) if labeled_differences else "",
            "median_frontier_aggregate_seconds": median(float(r["total_seconds"]) for r in frontier),
            "median_labeled_check_seconds": median(labeled_times) if labeled_times else "",
        }
    ]
    write_csv(TABLES / "triobjective_summary_generated.csv", summary)


def run_public() -> None:
    experiment = "public"
    log = reset_log(experiment)
    raw_rows: list[dict[str, object]] = []
    instance = public_instance()
    for formulation in ("aggregate", "labeled"):
        solution = solve_experimental_model(instance, formulation, time_limit=20.0, mip_rel_gap=1e-9)
        save_solution(log, raw_rows, experiment, instance, solution, stage="baseline")
        if formulation == "aggregate":
            if not solution.optimal:
                raise RuntimeError("public aggregate case was not certified")
            _, lifted_vector = lift_aggregate_solution(instance, solution)
            lift_difference = float(np.max(np.abs(lifted_vector - solution.objective_vector)))
            used = np.sum(solution.dispatch, axis=(0, 1)).astype(int)
            baseline_objective = float(solution.objective)
            baseline_used = used
    sensitivity_rows: list[dict[str, object]] = []
    for multiplier in (0.70, 0.85, 1.00, 1.15, 1.30, 1.50):
        flow_objectives = instance.flow_objectives.copy()
        flow_objectives[0, :, :, 0, 1] *= multiplier
        modified = ExperimentalInstance(
            **{**asdict(instance), "name": f"public_multiplier_{multiplier:.2f}", "flow_objectives": flow_objectives}
        )
        solution = solve_experimental_model(modified, "aggregate", time_limit=20.0, mip_rel_gap=1e-9)
        save_solution(
            log,
            raw_rows,
            experiment,
            modified,
            solution,
            stage="sensitivity",
            class_2_cost_multiplier=multiplier,
        )
        if not solution.optimal:
            raise RuntimeError(f"public sensitivity {multiplier} was not certified")
        used = np.sum(solution.dispatch, axis=(0, 1)).astype(int)
        sensitivity_rows.append(
            {
                "class_2_cost_multiplier": multiplier,
                "class_2_cost": 0.058 * multiplier,
                "objective": solution.objective,
                "class_1_vehicles": int(used[0]),
                "class_2_vehicles": int(used[1]),
            }
        )
    write_csv(RAW / "public_runs.csv", raw_rows)
    write_csv(TABLES / "public_sensitivity_generated.csv", sensitivity_rows)
    baseline_rows = [
        {
            "objective": baseline_objective,
            "class_1_vehicles": int(baseline_used[0]),
            "class_2_vehicles": int(baseline_used[1]),
            "lift_max_absolute_difference": lift_difference,
        }
    ]
    write_csv(TABLES / "public_baseline_generated.csv", baseline_rows)


def write_environment() -> None:
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "solver": "scipy.optimize.milp (HiGHS)",
        "main_seeds_by_family": {
            family: [seed + 100 * family_index for seed in MAIN_SEEDS]
            for family_index, (family, *_dimensions) in enumerate(MAIN_FAMILIES)
        },
        "clone_seed": CLONE_SEED,
        "robustness_seeds": ROBUSTNESS_SEEDS,
        "large_fleet_seeds": LARGE_FLEET_SEEDS,
        "triobjective_seed": TRIOBJECTIVE_SEED,
        "synthetic_supply_design": {
            "total_supply_factor": 1.10,
            "share_formula": "one seeded origin pair for all commodities: 0.85 dominant, 0.15 secondary",
            "guarantee": "no single origin can cover total commodity demand",
        },
        "time_limits_seconds": {
            "main": 5,
            "clone": 3,
            "robustness": 30,
            "large_fleet_milp": 15,
            "large_fleet_lp": 45,
            "triobjective": 20,
            "public": 20,
        },
    }
    (RESULTS / "environment.json").write_text(
        json.dumps(environment, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["all"],
        choices=("all", "main", "clone", "robustness", "large_fleet", "triobjective", "public"),
    )
    args = parser.parse_args()
    ensure_directories()
    write_environment()
    requested = set(args.experiments)
    if "all" in requested:
        requested = {"main", "clone", "robustness", "large_fleet", "triobjective", "public"}
    runners = {
        "main": run_main,
        "clone": run_clone,
        "robustness": run_robustness,
        "large_fleet": run_large_fleet,
        "triobjective": run_triobjective,
        "public": run_public,
    }
    for name in ("main", "clone", "robustness", "large_fleet", "triobjective", "public"):
        if name in requested:
            print(f"=== {name} ===", flush=True)
            runners[name]()
    print("requested experiments complete", flush=True)


if __name__ == "__main__":
    main()
