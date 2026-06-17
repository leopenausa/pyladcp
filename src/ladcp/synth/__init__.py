"""Forward-model synthetic LADCP dataset generator.

Produces a single dual-head station (down + up PD0 + cleaned CTD ``.cnv``) from a
*known* ocean velocity profile and seabed depth. The output runs end-to-end through
``ladcp-qa`` with no private dependencies, serving both as a public demo and as a
known-answer recovery test (see ``tests/test_synth_recovery.py``).
"""

from __future__ import annotations

from .dataset import SynthConfig, SynthPaths, generate_station
from .profile import OceanTruth, cast_trajectory, ctd_series, ocean_truth

__all__ = [
    "SynthConfig",
    "SynthPaths",
    "generate_station",
    "OceanTruth",
    "ocean_truth",
    "cast_trajectory",
    "ctd_series",
]
