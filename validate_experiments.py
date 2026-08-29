"""Validate the complete generated experimental evidence."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def ensure_results() -> None:
    """Restore the versioned evidence archive when validating a fresh clone."""
    if RESULTS.is_dir():
        return
    archive = ROOT / "reproducibility_results.zip"
    if not archive.is_file():
        raise FileNotFoundError(
            "results/ and reproducibility_results.zip are both missing"
        )
    root_resolved = ROOT.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (ROOT / member.filename).resolve()
            if target != root_resolved and root_resolved not in target.parents:
                raise ValueError(f"unsafe archive member: {member.filename}")
        bundle.extractall(ROOT)


def rows(relative: str) -> list[dict[str, str]]:
    with (RESULTS / relative).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def truth(value: str) -> bool:
    return value.lower() == "true"


def validate() -> dict[str, object]:
    ensure_results()
    report: dict[str, object] = {}

    main = rows("raw/main_runs.csv")
    assert len(main) == 36
    main_a = {(r["family"], r["seed"]): r for r in main if r["formulation"] == "A"}
    main_l = {(r["family"], r["seed"]): r for r in main if r["formulation"] == "L"}
    assert len(main_a) == len(main_l) == 18
    assert all(truth(r["optimal"]) for r in main_a.values())
    main_equal = [
        abs(float(main_a[key]["objective"]) - float(row["objective"]))
        for key, row in main_l.items()
        if truth(row["optimal"])
    ]
    assert max(main_equal) <= 1e-6
    report["main"] = {
        "runs": len(main),
        "aggregate_optima": sum(truth(r["optimal"]) for r in main_a.values()),
        "labeled_optima": sum(truth(r["optimal"]) for r in main_l.values()),
        "max_certified_objective_difference": max(main_equal),
    }

    clone = rows("raw/clone_runs.csv")
    assert len(clone) == 18
    clone_a = {r["vehicles_per_class"]: r for r in clone if r["formulation"] == "A"}
    clone_l = {r["vehicles_per_class"]: r for r in clone if r["formulation"] == "L"}
    clone_differences = [
        abs(float(clone_a[key]["objective"]) - float(row["objective"]))
        for key, row in clone_l.items()
    ]
    assert all(truth(r["optimal"]) for r in clone)
    assert max(clone_differences) <= 1e-6
    report["clone"] = {
        "runs": len(clone),
        "max_objective_difference": max(clone_differences),
    }

    robustness = rows("raw/robustness_runs.csv")
    assert len(robustness) == 90
    robust_a = {
        (r["vehicles_per_class"], r["seed"]): r
        for r in robustness
        if r["formulation"] == "A"
    }
    assert len(robust_a) == 30
    assert all(truth(r["optimal"]) for r in robust_a.values())
    robust_equal = []
    incumbent_gaps = []
    for row in robustness:
        if row["formulation"] == "A" or not row["objective"]:
            continue
        key = (row["vehicles_per_class"], row["seed"])
        reference = float(robust_a[key]["objective"])
        incumbent_gaps.append(100.0 * (float(row["objective"]) - reference) / reference)
        if truth(row["optimal"]):
            robust_equal.append(abs(float(row["objective"]) - reference))
    assert max(robust_equal) <= 1e-6
    report["robustness"] = {
        "runs": len(robustness),
        "aggregate_optima": 30,
        "labeled_optima": sum(
            truth(r["optimal"]) for r in robustness if r["formulation"] == "L"
        ),
        "symmetry_breaking_optima": sum(
            truth(r["optimal"]) for r in robustness if r["formulation"] == "SB"
        ),
        "max_certified_objective_difference": max(robust_equal),
        "max_incumbent_gap_percent": max(incumbent_gaps),
    }

    large = rows("raw/large_fleet_runs.csv")
    assert len(large) == 96
    key = lambda r: (r["family"], r["vehicles_per_class"], r["seed"])
    a_mip = {key(r): r for r in large if r["formulation"] == "A" and r["relaxation"] == "MILP"}
    a_lp = {key(r): r for r in large if r["formulation"] == "A" and r["relaxation"] == "LP"}
    l_lp = {key(r): r for r in large if r["formulation"] == "L" and r["relaxation"] == "LP"}
    assert len(a_mip) == len(a_lp) == len(l_lp) == 24
    assert all(truth(r["optimal"]) for r in a_mip.values())
    lp_differences = [
        abs(float(a_lp[k]["objective"]) - float(l_lp[k]["objective"]))
        for k in a_lp
        if a_lp[k]["objective"] and l_lp[k]["objective"]
    ]
    assert lp_differences
    assert max(lp_differences) <= 1e-8
    report["large_fleet"] = {
        "runs": len(large),
        "aggregate_milp_optima": 24,
        "labeled_milp_optima": sum(
            truth(r["optimal"])
            for r in large
            if r["formulation"] == "L" and r["relaxation"] == "MILP"
        ),
        "max_lp_objective_difference": max(lp_differences),
        "lp_objective_comparisons": len(lp_differences),
    }

    tri_summary = rows("tables/triobjective_summary_generated.csv")[0]
    tri_runs = rows("raw/triobjective_runs.csv")
    labeled_checks = [r for r in tri_runs if r.get("stage") == "labeled_check"]
    assert len(labeled_checks) == 10
    assert all(truth(r["optimal"]) for r in labeled_checks)
    assert float(tri_summary["max_lift_objective_difference"]) <= 1e-6
    assert float(tri_summary["max_labeled_primary_difference"]) <= 1e-6
    report["triobjective"] = {
        "sample_nondominated_vectors": int(tri_summary["sample_nondominated_vectors"]),
        "labeled_checks": len(labeled_checks),
        "max_lift_objective_difference": float(tri_summary["max_lift_objective_difference"]),
        "max_labeled_primary_difference": float(tri_summary["max_labeled_primary_difference"]),
    }

    public_runs = rows("raw/public_runs.csv")
    public_a = next(
        r
        for r in public_runs
        if r["formulation"] == "A" and r["stage"] == "baseline"
    )
    public_table = rows("tables/public_baseline_generated.csv")[0]
    assert truth(public_a["optimal"])
    assert abs(float(public_a["objective"]) - 66559.19642760069) <= 1e-6
    assert float(public_table["lift_max_absolute_difference"]) <= 1e-6
    assert len(rows("tables/public_sensitivity_generated.csv")) == 6
    report["public"] = {
        "objective": float(public_a["objective"]),
        "lift_max_absolute_difference": float(public_table["lift_max_absolute_difference"]),
    }

    report["validation"] = "PASS"
    (RESULTS / "validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    print(json.dumps(validate(), indent=2))
