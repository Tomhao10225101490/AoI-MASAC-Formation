# Reassemble metrics archives (Asia/Shanghai handoff)

From repo root (Git Bash / macOS / Linux):

```bash
cat results_archives/parts/maneuver_paper_suite_20261103_metrics.tar.xz.part* > results_archives/maneuver_paper_suite_20261103_metrics.tar.xz
cat results_archives/parts/v4_aux_and_v3_core_20261103_metrics.tar.xz.part* > results_archives/v4_aux_and_v3_core_20261103_metrics.tar.xz
tar -xJf results_archives/maneuver_paper_suite_20261103_metrics.tar.xz
tar -xJf results_archives/c_reshape_and_l0_20261103_metrics.tar.xz
tar -xJf results_archives/l0_continue_and_legacy_20261103_metrics.tar.xz
tar -xJf results_archives/v4_aux_and_v3_core_20261103_metrics.tar.xz
```

Also already on GitHub as whole files when push succeeds:
- results_archives/c_reshape_and_l0_20261103_metrics.tar.xz
- results_archives/l0_continue_and_legacy_20261103_metrics.tar.xz

Gate tips (no extract needed): results_key/
