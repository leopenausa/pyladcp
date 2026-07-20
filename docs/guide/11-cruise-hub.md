# 11 · The cruise hub — `ladcp`

One command takes you from "a directory of cruise data" to "processed, QA-scored
velocity profiles" — and then stays useful as the cruise grows. The hub has **two
front ends over one engine**: a guided browser window for anyone (this chapter's
first half), and terminal subcommands for SSH sessions and scripts (second half).
Both read the same `cruise.toml` and drive the same pipeline, so nothing you do in
one is invisible to the other.

## The window — `ladcp studio`

From your cruise directory:

```bash
ladcp studio
```

A browser window opens (local server, fully offline). What you get depends on where
the cruise stands:

- **No `cruise.toml` yet → the setup wizard.** The hub scans the directory
  (filenames only — nothing is read or copied) and shows what it found: your LADCP
  casts paired by name with their up-looker and CTD coverage, raw `.hex` casts with
  a from-hex conversion offer, every ship-ADCP candidate (VmDAS, CODAS, EK80) as an
  explicit choice, and nav tracks for the clock check. You confirm or correct each
  card, preview the exact `cruise.toml` it will write, save, and finish with a
  **trial station** — one cast processed on the spot with its QA scorecard inline
  and the PDF report one click away.
- **`cruise.toml` present → the cruise dashboard.** Three blocks in watch priority:
  what still needs processing (new/stale casts with reasons), how the processed
  casts scored (ok/warn/fail with the offending metrics and report links), and the
  loose ends (missing CTD, single-head casts, unconstrained solves, unapplied edit
  journals). *Process pending*, *process all*, or re-process one station — a
  progress bar tracks the run and the dashboard refreshes itself when it finishes.
- **`ladcp studio 07`** skips straight to the interactive single-station editor of
  [Chapter 10](10-studio.md) for that cast, with the cruise configuration already
  loaded.
- **EK80 on a remote share?** Choosing an EK80 source opens the extraction panel:
  the cast↔file timetable first (header peeks only), then — after your explicit
  confirmation — slim copies into `ek80/<station>/` with live progress, and
  `cruise.toml` re-pointed at the local copy. It is a *shallow* constraint
  (~15–140 m); [chapter 8](08-ship-adcp.md) says when it is worth it.

The window is a thin skin: every button presses the same code path as the terminal
commands below, and the setup wizard writes the same `cruise.toml` that `ladcp
init` does.

## The terminal — `ladcp init / status / process / config`

The same hub over SSH, in scripts, or when you just prefer text:

```bash
ladcp init            # guided first-run setup in the terminal (init --yes for scripts)
ladcp                 # bare: the status dashboard
ladcp process         # process everything new or stale (the freshness rule)
ladcp process 07 12   # named casts, unconditionally
ladcp process --all   # everything + the cruise-level exports
ladcp config show     # every option, annotated with where its value came from
```

Every wizard answer has a flag (`ladcp init --yes --root … --sadcp …`), so first-run
setup works over a flaky ship link too.

## `cruise.toml` — the cruise's one interface

Both front ends persist every decision in a single human-readable file at the
cruise root. Flags always win over the file; the file wins over built-in presets:

```toml
[cruise]
name = "MORIA2"

[data]
root = "."
out = "qa_out"

[sadcp]
folder = "sADCP/DATA"      # the constraint source you chose
timeoff = "auto"           # clock check against the nav track
nav = "nav"

[params]                   # cruise-wide CastParams overrides — no code edits, ever
pglim = 30.0

[params.MORIA-07]          # per-station layers win over [params]
zbottom = 3850.0
```

`ladcp-qa` and Studio read it automatically (`--no-config` opts out), so
`ladcp-qa 07 --dzbelow 24` means "the cruise configuration, plus this one change".
`ladcp config show` traces every resolved value to its source; `ladcp config edit`
opens `$EDITOR` and refuses to save a file that doesn't validate.

## The freshness rule

There is no hidden state: a cast is **done** when its QA report
(`qa_out/stations/<st>/<st>_qa.json`) is newer than every input — its PD0 heads,
CTD file, edit journal, and `cruise.toml` itself. `ladcp process` runs exactly
what's missing or stale, which means an interrupted batch resumes by rerunning it,
new casts are picked up on the next `ladcp process`, and editing `cruise.toml`
automatically marks every cast for reprocessing. `--force` overrides the rule.
