"""Comparison primitives for validating against golden references."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Check:
    name: str
    status: str                 # "pass" | "fail" | "pending"
    detail: str = ""
    value: Any = None
    reference: Any = None
    delta: Any = None
    tolerance: Any = None
    gate: bool = False          # if True, a failure fails the whole harness

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "gate": self.gate,
            "value": _j(self.value),
            "reference": _j(self.reference),
            "delta": _j(self.delta),
            "tolerance": _j(self.tolerance),
            "detail": self.detail,
        }


@dataclass
class Report:
    cast: str
    checks: list[Check] = field(default_factory=list)

    def add(self, c: Check) -> None:
        self.checks.append(c)

    @property
    def overall(self) -> str:
        gates = [c for c in self.checks if c.gate]
        if any(c.status == "fail" for c in self.checks):
            return "FAIL"
        if any(c.status == "pending" for c in gates):
            return "INCOMPLETE"
        if all(c.status == "pass" for c in gates) and gates:
            return "PASS"
        return "INCOMPLETE"

    def to_dict(self) -> dict:
        return {
            "cast": self.cast,
            "overall": self.overall,
            "checks": [c.to_dict() for c in self.checks],
        }


def scalar_check(name, value, reference, tol, *, rel=False, gate=False, unit="") -> Check:
    if value is None or reference is None or _isnan(value) or _isnan(reference):
        return Check(name, "pending", "missing value or reference",
                     value=value, reference=reference, tolerance=tol, gate=gate)
    delta = abs(value - reference)
    measure = delta / abs(reference) if rel and reference != 0 else delta
    status = "pass" if measure <= tol else "fail"
    kind = "rel" if rel else "abs"
    return Check(name, status,
                 f"|Δ|({kind})={measure:.5g} {unit} (tol {tol})".strip(),
                 value=value, reference=reference, delta=delta, tolerance=tol, gate=gate)


def profile_check(name, z, u, zref, uref, tol_median, *, gate=False) -> Check:
    """Interp golden onto python z-grid (or vice versa) and compare."""
    if u is None or uref is None or len(u) == 0 or len(uref) == 0:
        return Check(name, "pending", "profile not available", tolerance=tol_median, gate=gate)
    ref_on_z = np.interp(z, zref, uref, left=np.nan, right=np.nan)
    d = np.abs(u - ref_on_z)
    med = float(np.nanmedian(d))
    mx = float(np.nanmax(d))
    status = "pass" if med <= tol_median else "fail"
    return Check(name, status, f"median|Δ|={med:.4f}, max|Δ|={mx:.4f} m/s (tol {tol_median})",
                 value=med, reference=0.0, delta=med, tolerance=tol_median, gate=gate)


def _isnan(x) -> bool:
    try:
        return bool(np.isnan(x))
    except (TypeError, ValueError):
        return False


def _j(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x
