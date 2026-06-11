# Appendix D · Logsheet checklist

What to record at sea, per cast — the minimum set that makes every
[chapter 9](09-troubleshooting.md) diagnosis possible months later. Printable PDF
templates may be added here later; the *content* is what matters.

## LADCP cast log

| field | why it matters downstream |
|---|---|
| **Station label** — exactly as typed into the CTD console | becomes the cast's identity via the `.hex` anchor |
| Date, in-water / out-of-water times (UTC) | sanity-checks the index pairing |
| Deployment file name(s) per head (master/slave) | resolves `SHARED-MASTER` and missing-file questions |
| Echo-sounder depth at station | validates `bottom_depth` |
| Battery voltage before deployment | calibrates the WARN-only `battery` row |
| Instrument config changes (bin size, ping rate, coordinate frame) | explains sudden cross-cast differences |
| **Anything hung below the rosette** (corer, transponder, extra bottles) + its cable length | the near-field mask recipe needs the geometry ([chapter 7](07-solvers-weights.md)) |
| Winch stops, wire-outs, restarted recordings, late starts | explains `start_depth` / `profile_surface_coverage` / fragmented files |
| NMEA feed to the CTD deck unit verified? (once, at cruise start) | without it, **no automatic indexing is possible** |

## Ship-ADCP watch log (once per leg / on change)

| field | why |
|---|---|
| Instrument(s) + mode (e.g. OS150 narrowband) | CODAS `--sonar` parameter |
| VmDAS averaging interval (`.VMO`) | CODAS `--ens_len` |
| Acquisition PC clock vs GPS time — note any offset | the `--sadcp-timeoff` fix needs to know, even roughly |
| Raw single-ping recording (`.ENR`/`.N1R`/`.N2R`) on? | the best product needs it ([chapter 8](08-ship-adcp.md)) |
| Bubble/weather periods at full steam | explains underway data quality |

## The habit that pays

One line per cast, written *during* the cast, beats a perfect reconstruction
attempted at the end of the leg. Three of the validated processing pathologies in
this guide were confirmed from one-line logsheet notes ("monocorer hung at ~26 m",
"recording restarted", "NMEA cable swapped") that no amount of data analysis could
have established alone.
