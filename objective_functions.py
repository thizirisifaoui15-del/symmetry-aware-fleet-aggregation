"""Objective functions used by the aggregate and vehicle-labeled models.

For criterion q, the paper defines

    F_A[q] = sum(c[q] * X) + sum(f[q] * z)

for aggregate flows X and vehicle counts z, and

    F_L[q] = sum(c[q] * x) + sum(f[q] * y)

for vehicle-labeled flows x and dispatch decisions y.  Within each vehicle
class, c and f do not depend on the copy index.  Summing x and y over that
index therefore gives F_L = F_A component by component.

The vector functions below accept any number of additive criteria.  The first
axis of each coefficient array is the criterion axis.  Labeled decision arrays
use their final axis as the vehicle-copy axis; zero padding may be used when
classes contain different numbers of copies.
"""

from __future__ import annotations

import numpy as np


def _as_float_array(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains a non-finite value")
    return array


def aggregate_objective_vector(
    flows: np.ndarray,
    counts: np.ndarray,
    flow_coefficients: np.ndarray,
    dispatch_coefficients: np.ndarray,
) -> np.ndarray:
    """Return all additive criteria for an aggregate solution.

    ``flow_coefficients`` must have shape ``(p, *flows.shape)`` and
    ``dispatch_coefficients`` must have shape ``(p, *counts.shape)``.
    """

    flows = _as_float_array("flows", flows)
    counts = _as_float_array("counts", counts)
    flow_coefficients = _as_float_array("flow_coefficients", flow_coefficients)
    dispatch_coefficients = _as_float_array(
        "dispatch_coefficients", dispatch_coefficients
    )
    if flow_coefficients.ndim != flows.ndim + 1:
        raise ValueError("flow_coefficients must add one criterion axis")
    if dispatch_coefficients.ndim != counts.ndim + 1:
        raise ValueError("dispatch_coefficients must add one criterion axis")
    if flow_coefficients.shape[1:] != flows.shape:
        raise ValueError("flow coefficient and aggregate-flow shapes differ")
    if dispatch_coefficients.shape[1:] != counts.shape:
        raise ValueError("dispatch coefficient and vehicle-count shapes differ")
    if flow_coefficients.shape[0] != dispatch_coefficients.shape[0]:
        raise ValueError("flow and dispatch coefficients use different criteria")

    flow_axes = tuple(range(1, flow_coefficients.ndim))
    dispatch_axes = tuple(range(1, dispatch_coefficients.ndim))
    return np.sum(flow_coefficients * flows[None, ...], axis=flow_axes) + np.sum(
        dispatch_coefficients * counts[None, ...], axis=dispatch_axes
    )


def labeled_objective_vector(
    labeled_flows: np.ndarray,
    labeled_dispatch: np.ndarray,
    flow_coefficients: np.ndarray,
    dispatch_coefficients: np.ndarray,
) -> np.ndarray:
    """Return all criteria for a labeled solution with a final copy axis."""

    labeled_flows = _as_float_array("labeled_flows", labeled_flows)
    labeled_dispatch = _as_float_array("labeled_dispatch", labeled_dispatch)
    if labeled_flows.ndim < 1 or labeled_dispatch.ndim < 1:
        raise ValueError("labeled arrays must include a vehicle-copy axis")
    aggregate_flows = np.sum(labeled_flows, axis=-1)
    aggregate_counts = np.sum(labeled_dispatch, axis=-1)
    return aggregate_objective_vector(
        aggregate_flows,
        aggregate_counts,
        flow_coefficients,
        dispatch_coefficients,
    )


def aggregate_objective(
    flows: np.ndarray,
    counts: np.ndarray,
    flow_coefficients: np.ndarray,
    dispatch_coefficients: np.ndarray,
) -> float:
    """Return one aggregate objective value."""

    values = aggregate_objective_vector(
        flows,
        counts,
        np.asarray(flow_coefficients, dtype=float)[None, ...],
        np.asarray(dispatch_coefficients, dtype=float)[None, ...],
    )
    return float(values[0])


def labeled_objective(
    labeled_flows: np.ndarray,
    labeled_dispatch: np.ndarray,
    flow_coefficients: np.ndarray,
    dispatch_coefficients: np.ndarray,
) -> float:
    """Return one labeled objective value."""

    values = labeled_objective_vector(
        labeled_flows,
        labeled_dispatch,
        np.asarray(flow_coefficients, dtype=float)[None, ...],
        np.asarray(dispatch_coefficients, dtype=float)[None, ...],
    )
    return float(values[0])


def public_case_cost_coefficients(instance: object) -> tuple[np.ndarray, np.ndarray]:
    """Build the public-case flow and dispatch cost coefficient arrays."""

    from aggregate_transport import distance_matrix

    distances = distance_matrix(instance)
    ni = len(instance.origins)
    nj = len(instance.destinations)
    ng = len(instance.vehicle_classes)
    flow_coefficients = np.zeros((ni, nj, ng), dtype=float)
    dispatch_coefficients = np.zeros((ni, nj, ng), dtype=float)
    for i in range(ni):
        for j in range(nj):
            for g, vehicle in enumerate(instance.vehicle_classes):
                flow_coefficients[i, j, g] = (
                    vehicle.variable_cost_per_tonne_mile * distances[i, j]
                )
                dispatch_coefficients[i, j, g] = (
                    vehicle.dispatch_base_cost
                    + vehicle.dispatch_cost_per_mile * distances[i, j]
                )
    return flow_coefficients, dispatch_coefficients


def public_case_objective(
    instance: object, flows: np.ndarray, counts: np.ndarray
) -> float:
    """Independently recompute the scalar public-case economic objective."""

    flow_coefficients, dispatch_coefficients = public_case_cost_coefficients(
        instance
    )
    return aggregate_objective(
        flows, counts, flow_coefficients, dispatch_coefficients
    )
