# 7 · Solvers & weight tuning

!!! note "Draft pending"
    This chapter is planned but not yet written. Outline:

- The two solvers: full constrained inverse (default) vs shear method (`--solver shear`).
- The four weights and their physical meaning: `--botfac`, `--barofac`, `--sadcpfac`,
  `--smoofac` (1 = legacy balance, 0 = off, >1 = trust more).
- Reading the per-cast weights figure (`figures/<st>_weights.png`).
- `--down-only` as a cross-check tool.
- Pathologies and recipes: bad near-bottom bottom-track samples (`--botfac 0`),
  shallow shelf casts (`--dzbelow`), contaminated near-field bins.
