# Processing LADCP data with pyladcp

This guide takes you from a folder of raw lowered-ADCP (LADCP) files to
quality-assessed ocean velocity profiles, using
[pyladcp](https://github.com/leopenausa/pyladcp) — an open-source Python
implementation of the LDEO_IX inverse method
([Visbeck, 2002](https://doi.org/10.1175/1520-0426(2002)019%3C0794:DVPULA%3E2.0.CO;2)).

It is written in the spirit of the classic *"How to Process LADCP Data with the LDEO
Software"* manual: a practical, sea-going document. Theory appears only where you need
it to make a processing decision.

## How to use this guide

- **New to pyladcp?** Install ([chapter 3](03-installation.md)), then do the hands-on
  walkthrough ([chapter 4](04-first-station.md)) — it uses a real station that ships
  with the repository, so you need no data of your own.
- **Processing a cruise?** [Chapter 5](05-cruise-workflow.md) is the recipe;
  [chapter 6](06-qa-report.md) explains every number and figure in the reports it
  produces.
- **A cast looks wrong?** Go straight to [chapter 9 (troubleshooting)](09-troubleshooting.md)
  and [chapter 7 (solvers & weights)](07-solvers-weights.md).
- **Tuning a single station?** [Chapter 10](10-studio.md) is the interactive GUI
  (`ladcp-studio`): move the weights and watch the profile respond live, then copy the
  `ladcp-qa` command that reproduces what you found.
- **New to pyladcp, or starting a cruise?** [Chapter 11](11-cruise-hub.md) is the
  cruise hub: `ladcp studio` opens a window that finds your data, sets the cruise up
  step by step, and processes it — `ladcp init/status/process` are the same hub in
  the terminal.

Every command shown in this guide is executed by pyladcp's continuous-integration tests
against the repository's built-in test station — if the guide and the code ever
disagree, the build fails. What you read here is what the code does.

## Citing

If pyladcp contributes to your work, please cite the package and Visbeck (2002) — see
the [README](https://github.com/leopenausa/pyladcp#citing--acknowledging-pyladcp) for
ready-made citations and a suggested acknowledgment sentence.

**Contact:** Leopoldo D. Pena · Universitat de Barcelona ·
[lpena@ub.edu](mailto:lpena@ub.edu) ·
[ORCID 0000-0001-6414-6293](https://orcid.org/0000-0001-6414-6293)
