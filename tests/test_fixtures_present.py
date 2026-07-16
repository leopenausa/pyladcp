"""Guard against the green-bar illusion on partial checkouts.

~25 test modules skip themselves when the committed MORIA golden fixture is absent,
so a checkout without it passes with half the suite silently gone. CI sets
``LADCP_REQUIRE_GOLDEN=1``, turning a missing fixture into a hard failure there while
local partial checkouts keep their skip behavior.
"""
import os
from pathlib import Path

import pytest

_DOWN = (Path(__file__).parent / "fixtures" / "New_golden" / "Good" / "LADCP"
         / "MORIA-80-LADCP-M.000")


def test_golden_fixture_present_when_required():
    if not os.environ.get("LADCP_REQUIRE_GOLDEN"):
        pytest.skip("LADCP_REQUIRE_GOLDEN not set — partial checkouts may skip")
    assert _DOWN.is_file(), (
        f"golden fixture missing: {_DOWN} — the MORIA-gated test modules would all "
        "silently skip, leaving a hollow green bar")
