# 10 · Interactive Studio

`ladcp-studio` is a local web GUI for **single-station, interactive** processing: the
velocity profile re-renders live while you move the solver weights, toggle constraints,
or change the editing options. It runs the exact same engine as `ladcp-qa` — the
results are bit-identical, enforced by the test suite — so anything you find in Studio,
`ladcp-qa` reproduces.

Use it for the decisions chapter 7 describes: *should this cast get `--down-only`? does
the SADCP constraint fight the bottom track here? what does `smoofac` actually cost
me?* Instead of a re-run per question, you watch the answer continuously.

![pyladcp Studio solving MORIA-80](assets/studio_moria80.png)

## Install & launch

The GUI dependencies (FastAPI + uvicorn) are an optional extra:

```bash
pip install -e ".[gui]"
```

Launch it the same way you would call `ladcp-qa` — station ids plus the discovery
context. With the repository's built-in test station:

```bash
ladcp-studio 80 --root tests/fixtures/New_golden/Good
```

The browser opens by itself (suppress with `--no-browser`; the server is
localhost-only). Everything you know from `ladcp-qa` carries over:

```bash
ladcp-studio 80 79 --root "$B" --cruise MYCRUISE          # several stations, ‹ › to switch
ladcp-studio --index "$B/.ladcp_archive.json" --root "$B" # every cast in the archive index
ladcp-studio 80 --sadcp "$B/sADCP/DATA"                   # enable the ship-ADCP constraint
ladcp-studio 80 --sadcp "$B/sADCP/DATA" \
             --sadcp-codas "$B/codas/os150nb_enr"         # raw AND a CODAS product
```

`--sadcp` works exactly as in `ladcp-qa` (chapter 8) and is validated at launch — if
the folder doesn't directly hold the `.STA` files, the error tells you which subfolder
does.

The **SADCP constraint toggle** turns the ship-ADCP rows on and off; the **source
dropdown** under it picks which product feeds them. It lists every `--sadcp` folder
(repeatable — e.g. the 75 and 150 kHz instruments), every `--sadcp-codas` product
(chapter 8 / the CODAS guide), and any CODAS products **found automatically** under
the conventional `<root>/codas/` — those are marked *(found)* and never become the
default constraint; selecting one is your explicit choice. Pin a solution, switch
the source, and the Δ-strip shows what the instrument or the processing chain is
worth at this station. Each choice is still a single `ladcp-qa` command
(`--sadcp <path> [--sadcp-source codas]`), shown in the CLI bar as always. The
full flag list:

<!-- guide-test -->
```bash
ladcp-studio --help
```

## The screen

- **Left rail** — the processing state. *Pipeline* shows per-stage timings; *Solver*
  is `shear`/`inverse` plus the four weights of chapter 7; *Constraints* toggles
  `--down-only` and the SADCP rows; *Editing* exposes the near-field mask (a toggle,
  off by default — nothing is masked unless you opt in; the input takes bins `3,4`
  or a depth range `22-38m`, translated with the station's real bin geometry) and
  `dzbelow`. Hover any control for a condensed explanation.
- **Center** — the live u/v profile with its ±1σ band, bottom-track points, the
  ship-ADCP constraint (white squares) when active, and the seabed line. The bar
  underneath always shows **the `ladcp-qa` command that reproduces the current
  state** — `copy CLI` puts it on the clipboard.
- **Right** — your manual edits (the brush journal, see below), pinned solutions, and
  the QA panels (the same matplotlib figures as the `ladcp-qa` report, rendered on
  demand for the *current* configuration).

## The two speeds

Controls are grouped by what they cost, and the grouping is the point:

| tier | controls | cost |
|---|---|---|
| solve | solver, `botfac`, `barofac`, `smoofac`, `sadcpfac`, SADCP toggle | **~30 ms** — drag and watch |
| build | `--down-only`, near-field bins, `dzbelow`, each brush edit | **~1.5 s** — rebuilds editing → bottom detect → super-ensembles |

The first visit to a station pays raw ingest + build once (~1–2 s on the test
station); after that, weight changes are effectively instantaneous.

## Brush editing — the ✏ edit view

The `profile | ✏ edit` switch above the plot opens the **raw ensemble matrix**: one
head at a time (DN/UP), every bin against every ensemble of the cast, coloured by
**|error velocity|** (the natural QC field on earth-coordinate data) or **echo
amplitude** (where a hung device or interference is often most obvious to the eye).
Grey cells are what the automatic screening already removed — you never need to
re-flag those, and a brush can never bring them back.

![the brush edit view: the hung-device band flagged on bins 3–4](assets/studio_edit_moria80.png)

**Drag a rectangle** over bad cells to flag them. The rectangle becomes an entry in
the **Manual edits** card, the profile re-solves immediately (a brush is a build-tier
change, ~1.5 s), and the ✕ on an entry removes it again — the solution returns
**bit-identically** to what it was. A brush wider than ~90 % of the cast snaps to
*all ensembles*, i.e. a pure bin mask: flagging bins 3–4 across the whole cast is
exactly `--nearfield-dn-bins 3,4`, bit for bit.

This is the tool for the artifacts thresholds can't catch: a rigid device hung below
the package reads the *package's* motion at |errvel| well under the 0.2 m/s edit
limit (chapter 6) — coherent, biased, and invisible to a global threshold, but
obvious as a fixed-range band on this screen. Same for wake bursts while the ship
holds station, or another instrument's interference stripes.

### The journal: recorded, replayable, never silent

Brush edits persist to a per-station **journal** —
`<root>/.ladcp_edits/<station>.json` — and that file is the single source of truth:
it records each rectangle's geometry, your note, and fingerprints of the raw files
it was drawn on. Nothing else changes on disk, and nothing is ever applied silently:

- In batch, the journal applies **only** with an explicit flag —
  `ladcp-qa 80 --edits .ladcp_edits/` (the directory, per-station lookup) or
  `--edits .ladcp_edits/MORIA-80.json` (one file, one station). The CLI bar shows
  this command whenever edits are active, and the replay is bit-identical to the
  Studio solution.
- When edits **are** applied, the QA report says so (`manual_edits`, with the count
  and the journal path). When a journal exists but you ran *without* `--edits`, the
  report WARNs (`manual_edits_unapplied`) and names the exact flag to add — the same
  actionable-note contract as the hung-device detector.
- If a raw file changes after the edits were drawn (re-downloaded, re-cut), the
  fingerprints catch it: the journal is refused with a clear message rather than
  letting the rectangles land on the wrong cells. Delete or re-create the journal.

Hand-edit the JSON if you like (notes, removing entries) — bins are 1-based per
head, ensembles 0-based; anything the loader doesn't fully understand is refused,
never skipped.

## A workflow that works

1. Solve with defaults, press **⊕ pin** — that's your baseline, frozen as a dashed
   ghost.
2. Change *one thing* (chapter 7's rule applies in Studio too). The **Δ-strip** under
   the profile shows live − pin against depth, so a 2 cm/s shift confined to the upper
   300 m reads at a glance.
3. Run the standard cross-checks without leaving the page: `shear` vs `inverse`,
   dual vs `--down-only` — pin one, switch, read the Δ.
4. Found a configuration you trust? **copy CLI** and run it in your batch script, or
   **↓ .lad** to download the profile directly (byte-identical to the `ladcp-qa`
   output).

Studio is a *diagnosis* tool: it holds one station at a time (the three most recent
stay cached) and the only thing it writes to disk is the edit journal you brush.
Production outputs — reports, exports, the cruise aggregate — stay with `ladcp-qa`
(chapter 5), fed by the command line Studio hands you.
