# Local artifacts not uploaded to GitHub (2026-11-04 Asia/Shanghai)

TomMac-mini path: `/Users/a1/Desktop/research/MASAC`

Total `results/` on disk ≈ **43 GB**. GitHub upload keeps **research metrics/text** for `masac_v3*` / `masac_v4*` / `v4_*` (json/jsonl/log/md/csv/png/…).

## Left local (by reason)

| Reason | Size |
|---|---:|
| `replay_buffer_npz` | 27.57 GB |
| `model_weights_pt` | 12.90 GB |
| `non_masac_research_tree` | 4.52 GB |
| `eval_trajectory_npz` | 0.49 GB |
| `binary_.npz` | 0.02 GB |
| `unknown_large_nontext` | 0.00 GB |
| `nontext_ext_.html` | 0.00 GB |
| `junk_lock_or_cache` | 0.00 GB |

## Largest local dirs (full tree, including uploaded metrics)

| Dir | Size |
|---|---:|
| `results/masac_v4_maneuver_paper_suite_20261103_zero_20261103_162345` | 4.98 GB |
| `results/masac_v4_maneuver_paper_suite_20261103_full_20261103_162345` | 4.98 GB |
| `results/masac_v4_maneuver_paper_suite_20261103_static_20261103_162345` | 4.97 GB |
| `results/masac_v3_paired_300k_20261103` | 4.41 GB |
| `results/day` | 3.58 GB |
| `results/study` | 2.28 GB |
| `results/masac_v4_l0_legacy_50k_20261103_102637` | 1.89 GB |
| `results/masac_v4_c_reshape_l0_frozen42_20261103_145043` | 1.88 GB |
| `results/masac_v4_l0_continue_522_plus50k_20261103_122256` | 1.62 GB |
| `results/masac_v4_l0_continue_511_plus50k_20261103_122254` | 1.61 GB |
| `results/masac_v4_l0_continue_533_plus50k_20261103_121717` | 1.61 GB |
| `results/dc_rrt_pruning_confirm5000_20261102` | 1.18 GB |
| `results/dc_rrt_pruning_20261102` | 0.67 GB |
| `results/local_pilot_20261031` | 0.65 GB |
| `results/story_residual` | 0.65 GB |
| `results/story` | 0.65 GB |
| `results/masac_v3_goal_curriculum_20261102` | 0.61 GB |
| `results/masac_v3_budget_20261102` | 0.58 GB |
| `results/dc_rrt_pruning_final_20261102` | 0.55 GB |
| `results/masac_planning_pilot_v2_20261102` | 0.47 GB |
| `results/masac_planning_pilot_20261102` | 0.46 GB |
| `results/masac_v4_c_reshape_empty_20261103_142623` | 0.41 GB |
| `results/dc_rrt_pruning_confirm1000_20261102` | 0.39 GB |
| `results/candidate_c_pilot_20261031_v2` | 0.37 GB |
| `results/candidate_c_pilot_20261031` | 0.37 GB |
| `results/dc_rrt_fidelity_full_20261102` | 0.36 GB |
| `results/masac_v4_critic_sampling_uniform_411_20261103_122320` | 0.33 GB |
| `results/dc_rrt_20261102_evaluation` | 0.32 GB |
| `results/v4_critic_sampling_smoke_v2_20261103` | 0.32 GB |
| `results/v4_critic_sampling_smoke_20261103` | 0.32 GB |
| `results/dc_rrt_intensity_eval_20261103_122351` | 0.32 GB |
| `results/overnight_20261101` | 0.23 GB |
| `results/masac_v3_baseline_20261102` | 0.22 GB |
| `results/v4_critic_sampling_recovery_smoke_20261103` | 0.16 GB |
| `results/masac_planning_baseline_budget_20261102` | 0.15 GB |
| `results/dc_rrt_pruning_density1000_20261102` | 0.14 GB |
| `results/dc_rrt_pruning_density5000_20261102` | 0.14 GB |
| `results/masac_v4_capture_latch_off_20261103_132438` | 0.13 GB |
| `results/masac_v4_capture_latch_on_20261103_131315` | 0.13 GB |
| `results/masac_v4_sanity_empty_single_20261103_131315` | 0.13 GB |
| `results/upgrade_audit` | 0.06 GB |
| `results/dc_rrt_goal_threshold_20261102` | 0.06 GB |

## How to get weights/replays on another machine

- Stay on Mac mini for resume-from-checkpoint, **or**
- Copy selectively via AirDrop/rsync/USB, e.g. final `checkpoint_080000/checkpoint.pt` per seed (~25 MB each).
- Do **not** commit `*.pt` / `replay*.npz` (GitHub 100 MB file limit; total ~41 GB).

## Uploaded research metrics scope

- Files staged: **15904** (~133.5 MB)
- Trees: `results/masac_v3*`, `results/masac_v4*`, `results/v4_*`, plus matching root sidecars.
- Includes: `completion.json`, `status.json`, eval `*_map.json`, train logs, launcher.out, configs snapshots, gate diagnosis folders.

## Uploaded via archives

See `results_archives/*_metrics.tar.xz` and `docs/RESULTS_ARCHIVES_README_20261104.md`. Plaintext gate tips also in `results_key/`.
