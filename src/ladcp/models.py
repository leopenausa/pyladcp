"""Data-contract objects (see docs/ARCHITECTURE.md).

These are deliberately lightweight dataclasses / containers. Heavy gridded products
also know how to serialise to the legacy `.lad`/`.bot` ASCII formats (see qa/export.py),
but those are added incrementally.

Conventions: SI units. Horizontal velocity carried as separate ``u`` (east) and ``v``
(north) arrays. Depth ``z`` positive-down in metres. Time as UTC ``datetime64[ns]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class Status(str, Enum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"


class CoordFrame(str, Enum):
    BEAM = "beam"
    INSTRUMENT = "instrument"
    SHIP = "ship"
    EARTH = "earth"


@dataclass
class RawADCP:
    """Decoded single-head PD0 data (one ensemble per column).

    Velocity array semantics depend on ``coord_frame``: for EARTH, the 4 components
    are (east, north, up, error); for BEAM, they are (b1, b2, b3, b4).
    """

    head: str                       # "master"/"downlooker" or "slave"/"uplooker"
    coord_frame: CoordFrame
    facing: str                     # "down" | "up"
    freq_khz: int
    n_beams: int
    n_cells: int
    cell_m: float                   # depth cell length [m]
    blank_m: float                  # blank after transmit [m]
    wm_mode: int                    # water-profiling mode (15 = LADCP)
    pings_per_ens: int
    serial: int | float | None

    time: np.ndarray                # [t] datetime64[ns] UTC
    vel: np.ndarray                 # [4, ncell, t] m/s (NaN = bad)
    echo: np.ndarray                # [4, ncell, t] dB-counts
    corr: np.ndarray                # [4, ncell, t] counts
    pct_good: np.ndarray            # [4, ncell, t] %
    heading: np.ndarray             # [t] deg
    pitch: np.ndarray               # [t] deg
    roll: np.ndarray                # [t] deg
    temperature: np.ndarray         # [t] deg C
    sound_speed: np.ndarray         # [t] m/s
    salinity: np.ndarray            # [t] PSU
    xmit_voltage: np.ndarray | None  # [t] engineering units (battery proxy)

    bt_range: np.ndarray | None     # [4, t] m  (per-beam bottom range)
    bt_vel: np.ndarray | None       # [4, t] m/s (bottom-track velocity, same frame as vel)

    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_ens(self) -> int:
        return self.time.size


@dataclass
class CTDTimeSeries:
    """1 Hz CTD time-series with merged GPS (the cleaned `.cnv`)."""

    time_elapsed_s: np.ndarray      # [t] seconds since cast start
    lat: np.ndarray                 # [t] deg N
    lon: np.ndarray                 # [t] deg E
    pressure: np.ndarray            # [t] dbar
    temperature: np.ndarray         # [t] deg C
    salinity: np.ndarray            # [t] PSU
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return self.time_elapsed_s.size


@dataclass
class Metric:
    name: str
    value: Any
    unit: str = ""
    status: Status = Status.OK
    threshold: Any = None
    source_stage: str = ""
    note: str = ""


@dataclass
class QCMetrics:
    """Acquisition-quality assessment object (the priority deliverable)."""

    station: str
    metrics: dict[str, Metric] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def add(self, metric: Metric) -> None:
        self.metrics[metric.name] = metric

    @property
    def overall_status(self) -> Status:
        if any(m.status == Status.FAIL for m in self.metrics.values()):
            return Status.FAIL
        if any(m.status == Status.WARN for m in self.metrics.values()):
            return Status.WARN
        return Status.OK

    def to_dict(self) -> dict[str, Any]:
        return {
            "station": self.station,
            "overall_status": self.overall_status.value,
            "warnings": self.warnings,
            "metrics": {
                k: {
                    "value": _jsonable(m.value),
                    "unit": m.unit,
                    "status": m.status.value,
                    "threshold": _jsonable(m.threshold),
                    "source_stage": m.source_stage,
                    "note": m.note,
                }
                for k, m in self.metrics.items()
            },
        }


def _jsonable(x: Any) -> Any:
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x
