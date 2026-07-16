"""Readers and writers for every on-disk format the pipeline touches.

LADCP:  raw RDI PD0 (``pd0``, with ``beam2earth`` rotation), PD0 writing for the
synthetic generator (``pd0_write``).
CTD:    cleaned ``.cnv`` time series (``ctd_cnv``), raw ``.hex`` header anchors
(``ctd_hex``), on-the-fly ``.hex`` -> ``.cnv`` conversion (``ctd_raw``).
Ship-ADCP (all return a ``sadcp_types.SadcpDataset``): raw VmDAS (``sadcp_vmdas``),
CODAS-processed NetCDF (``sadcp_codas``), Simrad EK80 ADCP-mode (``sadcp_ek80``, with
``ek80_files`` timetable/slimming tools and shared ``ek80_common`` helpers).
Navigation: ship GPS tracks (``nav``).
"""
