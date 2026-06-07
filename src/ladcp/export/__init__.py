"""Shareable-output writers for the velocity solution (roadmap "Outputs", #3).

The LDEO ``.lad``/``.bot`` text writers stay in :mod:`ladcp.qa.export`; this subpackage
adds the formats oceanographers exchange — an Ocean Data View Generic Spreadsheet
(:mod:`~ladcp.export.odv`), CF-style NetCDF (:mod:`~ladcp.export.netcdf`), and Excel
workbooks (:mod:`~ladcp.export.xlsx`) — all built from the one tidy-table core in
:mod:`~ladcp.export.tables` so the columns/units stay consistent across formats.

Excel needs the optional ``openpyxl`` dependency (``pip install pyladcp[export]``); the
writer raises :class:`ExportDependencyError` when it is missing so callers can warn and
skip without aborting the other formats.
"""

from __future__ import annotations


class ExportDependencyError(RuntimeError):
    """An optional export dependency (e.g. ``openpyxl`` for Excel) is not installed."""


from .tables import StationExport  # noqa: E402  (re-export after the error class)

__all__ = ["ExportDependencyError", "StationExport"]
