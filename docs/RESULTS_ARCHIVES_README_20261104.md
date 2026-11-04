# Results metrics archives (2026-11-04 Asia/Shanghai)

GitHub HTTPS pushes of thousands of small `results/**/*.json` files were timing out (HTTP 408).
Research metrics are therefore shipped as xz archives under `results_archives/`.

## Extract on Windows (PowerShell or Git Bash)
```bash
tar -xJf results_archives/maneuver_paper_suite_20261103_metrics.tar.xz
tar -xJf results_archives/c_reshape_and_l0_20261103_metrics.tar.xz
tar -xJf results_archives/l0_continue_and_legacy_20261103_metrics.tar.xz
tar -xJf results_archives/v4_aux_and_v3_core_20261103_metrics.tar.xz
```
Archives expand to `results/...` relative to repo root.

## Still local only on Mac mini (~41 GB)
See `docs/LOCAL_ARTIFACTS_NOT_ON_GITHUB_20261104.md` — `*.pt` checkpoints, `replay*.npz`, `eval_*.npz`, and non-MASAC trees (dc_rrt/day/…).
