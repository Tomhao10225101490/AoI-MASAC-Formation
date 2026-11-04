# Results metrics archives (2026-11-04 Asia/Shanghai)

GitHub HTTPS pushes of multi-MB binaries often time out (HTTP 408) from TomMac-mini.
Research metrics are therefore shipped under `results_archives/`:

1. Whole xz archives already on GitHub:
   - `c_reshape_and_l0_20261103_metrics.tar.xz`
   - `l0_continue_and_legacy_20261103_metrics.tar.xz`
2. Split parts (≤350 KiB) for the remaining two archives under `results_archives/parts/`:
   - `maneuver_paper_suite_20261103_metrics.tar.xz.partaa` … `partag`
   - `v4_aux_and_v3_core_20261103_metrics.tar.xz.partaa` … `partau`

Plaintext gate tips (no extract): `results_key/` (empty / L0 / full / static / zero).

## Extract on Windows (PowerShell or Git Bash)

```bash
# Reassemble split archives
cat results_archives/parts/maneuver_paper_suite_20261103_metrics.tar.xz.part* > results_archives/maneuver_paper_suite_20261103_metrics.tar.xz
cat results_archives/parts/v4_aux_and_v3_core_20261103_metrics.tar.xz.part* > results_archives/v4_aux_and_v3_core_20261103_metrics.tar.xz

# Expand all four
tar -xJf results_archives/maneuver_paper_suite_20261103_metrics.tar.xz
tar -xJf results_archives/c_reshape_and_l0_20261103_metrics.tar.xz
tar -xJf results_archives/l0_continue_and_legacy_20261103_metrics.tar.xz
tar -xJf results_archives/v4_aux_and_v3_core_20261103_metrics.tar.xz
```

Archives expand to `results/...` relative to repo root. See also `results_archives/parts/README_REASSEMBLE.txt`.

## Still local only on Mac mini (~41 GB)

See `docs/LOCAL_ARTIFACTS_NOT_ON_GITHUB_20261104.md` — `*.pt` checkpoints, `replay*.npz`, `eval_*.npz`, and non-MASAC trees (dc_rrt/day/…).
