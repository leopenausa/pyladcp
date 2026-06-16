"""Roadmap #3 — the pure tidy-table builders (single column/unit source of truth)."""

from __future__ import annotations

import numpy as np

from ladcp.export.tables import (
    bottomtrack_frame,
    locate_frame,
    metadata_dict,
    profile_frame,
    qa_frame,
    sadcp_frame,
    shear_frame,
    summary_row,
)


def test_profile_frame_columns_and_nan_drop(synth_export):
    df = profile_frame(synth_export.result)
    assert list(df.columns) == ["depth_m", "u_ms", "v_ms", "uerr_ms", "n_vel"]
    # the one all-NaN velocity bin (index 2) is dropped, like write_lad
    assert np.isfinite(df["u_ms"]).all() and np.isfinite(df["v_ms"]).all()
    assert len(df) == 19          # 20 bins minus the NaN row


def test_bottomtrack_and_shear_frames(synth_export):
    bt = bottomtrack_frame(synth_export.result)
    assert list(bt.columns) == ["depth_m", "u_ms", "v_ms", "uerr_ms", "n"]
    sh = shear_frame(synth_export.result)
    assert list(sh.columns) == ["depth_m", "u_ms", "v_ms", "u_shear_1s", "v_shear_1s", "n"]


def test_sadcp_frame_optional(synth_export):
    sv = sadcp_frame(synth_export.result)
    assert list(sv.columns) == ["depth_m", "u_ms", "v_ms", "verr_ms"]
    # a station without a ship-ADCP constraint yields None
    synth_export.result.sadcp = None
    assert sadcp_frame(synth_export.result) is None


def test_locate_frame_prepends_position(synth_export):
    df = profile_frame(synth_export.result)
    located = locate_frame(df, synth_export)
    assert list(located.columns[:3]) == ["station", "latitude_deg", "longitude_deg"]
    assert (located["station"] == "MORIA-80").all()
    assert (located["latitude_deg"] == synth_export.lat).all()
    assert (located["longitude_deg"] == synth_export.lon).all()
    # original frame is untouched and the velocity columns survive after the prefix
    assert "station" not in df.columns
    assert list(located.columns[3:]) == list(df.columns)


def test_qa_frame_rows(synth_export):
    qa = qa_frame(synth_export.qc)
    assert list(qa.columns) == ["metric", "value", "unit", "status", "threshold", "note"]
    assert set(qa["metric"]) == {"battery", "tilt_max"}
    assert qa.loc[qa["metric"] == "battery", "status"].iloc[0] == "warn"


def test_metadata_and_summary(synth_export):
    md = metadata_dict(synth_export)
    assert md["station"] == "MORIA-80"
    assert md["solver"] == "inverse"
    assert md["bottom_depth_m"] == 1057.0
    assert md["n_bins"] == 19                 # finite velocity bins
    assert "pyladcp_version" in md
    assert md["sadcp_source"] == "sADCP/sadcp_150/DATA"

    row = summary_row(synth_export)
    assert set(row) == {"station", "latitude_deg", "longitude_deg", "time_utc",
                        "overall_status", "ubar_ms", "vbar_ms", "bottom_depth_m", "n_bins",
                        "sadcp_independent_rms_ms"}
    assert row["overall_status"] == "warn"
