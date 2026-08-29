"""Recreate the public-case input tables from observed and calibrated inputs."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def read_csv(name: str) -> list[dict[str, str]]:
    with (ROOT / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(name: str, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with (ROOT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def calibrate() -> None:
    places = read_csv("public_us_places_source.csv")
    calibration = read_csv("public_case_calibration.csv")
    scenario = next(row for row in calibration if row["record_type"] == "scenario")
    total_demand = float(scenario["total_demand_tonnes"])

    stock = {
        row["name"]: float(row["stock_tonnes"])
        for row in calibration
        if row["record_type"] == "origin"
    }
    origins = [row for row in places if row["role"] == "origin"]
    origin_rows = [
        {
            "origin": row["place"],
            "latitude": row["latitude"],
            "longitude": row["longitude"],
            "stock_tonnes": f"{stock[row['place']]:g}",
            "data_status": "observed_coordinates_calibrated_stock",
        }
        for row in origins
    ]

    destinations = [row for row in places if row["role"] == "destination"]
    population_total = sum(int(row["population_2025"]) for row in destinations)
    destination_rows = [
        {
            "destination": row["place"],
            "latitude": row["latitude"],
            "longitude": row["longitude"],
            "demand_tonnes": f"{total_demand * int(row['population_2025']) / population_total:.9f}",
            "data_status": "observed_coordinates_population_weighted_demand",
        }
        for row in destinations
    ]

    vehicle_rows = [
        {
            "vehicle_class": row["name"],
            "capacity_tonnes": row["capacity_tonnes"],
            "vehicles": row["vehicles"],
            "variable_cost_per_tonne_mile": row["variable_cost_per_tonne_mile"],
            "dispatch_base_cost": row["dispatch_base_cost"],
            "dispatch_cost_per_mile": row["dispatch_cost_per_mile"],
            "data_status": "calibrated_scenario_assumption",
        }
        for row in calibration
        if row["record_type"] == "vehicle"
    ]

    write_csv(
        "public_us_origins.csv",
        ["origin", "latitude", "longitude", "stock_tonnes", "data_status"],
        origin_rows,
    )
    write_csv(
        "public_us_destinations.csv",
        ["destination", "latitude", "longitude", "demand_tonnes", "data_status"],
        destination_rows,
    )
    write_csv(
        "vehicle_classes.csv",
        [
            "vehicle_class",
            "capacity_tonnes",
            "vehicles",
            "variable_cost_per_tonne_mile",
            "dispatch_base_cost",
            "dispatch_cost_per_mile",
            "data_status",
        ],
        vehicle_rows,
    )


if __name__ == "__main__":
    calibrate()
    print("Public-case input tables recreated.")
