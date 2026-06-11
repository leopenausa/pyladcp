"""pyladcp Studio: a local web GUI for interactive single-station processing.

A thin FastAPI layer over :class:`ladcp.session.StationSession` -- the GUI can never
produce a solution the CLI cannot reproduce (every solve response carries its
``ladcp-qa`` command line). Install the extra and launch::

    pip install "pyladcp[gui]"
    ladcp-studio 80 --root New_golden/Good --cruise MORIA
"""
