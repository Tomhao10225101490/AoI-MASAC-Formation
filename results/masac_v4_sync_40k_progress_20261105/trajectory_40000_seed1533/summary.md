# Post-hoc trajectory diagnosis: seed 1533, step 40000

Post-hoc intermediate diagnosis only. The registered final40k primary endpoint and six-history advancement rule remain unchanged; no training change, automatic extension or sealed-test evaluation is authorized.

Each arm has 36 correlated agent-map observations from one training seed and 12 development maps; these are descriptive counts, not 36 independent trials.

Capture occupancy counts position/speed only. Success remains collision-free joint capture; all four nonexclusive collision flags are retained per map and per arm in summary.json.

min_distance_m and JSON closest include reset frame, matching evaluate(); capture/ever-ball/streak/timing use only post-step frames.

Range of first-capture times only if every agent ever captures; null otherwise. Eventual capture and first-arrival gap do not establish simultaneous capture.

| Arm | Never ball | Ball without capture | Ever capture | Success maps | Any-ball maps | All-ball maps | All-ever-capture maps |
|---|---:|---:|---:|---:|---:|---:|---:|
| control | 11 | 0 | 25 | 4/12 | 12/12 | 5/12 | 5/12 |
| treatment | 4 | 3 | 29 | 5/12 | 12/12 | 8/12 | 7/12 |

## control: map descriptions

| Map seed | Family | Never / ball-only / captured agents | Max occupancy | Worst closest m | Worst final m | Worst final speed m/s | First-capture gap s |
|---|---|---|---:|---:|---:|---:|---|
| 830001 | urban_canyon | 2 / 0 / 1 | 1 | 5.3579 | 7.1224 | 0.3405 | None |
| 830002 | warehouse | 1 / 0 / 2 | 2 | 5.2535 | 12.7570 | 0.1460 | None |
| 830003 | courtyard | 2 / 0 / 1 | 1 | 5.1635 | 20.1976 | 0.1094 | None |
| 830004 | overpass | 0 / 0 / 3 | 3 | 3.8941 | 3.8941 | 1.0724 | 6.6000 |
| 830005 | narrow_crossing | 0 / 0 / 3 | 3 | 3.9067 | 3.9067 | 1.7559 | 1.6000 |
| 830006 | dynamic_crossing | 1 / 0 / 2 | 2 | 4.2739 | 6.1500 | 0.1912 | None |
| 830007 | urban_canyon | 0 / 0 / 3 | 3 | 3.9146 | 3.9146 | 0.8114 | 4.6000 |
| 830008 | warehouse | 0 / 0 / 3 | 3 | 3.8535 | 3.8535 | 1.2902 | 1.0000 |
| 830009 | courtyard | 0 / 0 / 3 | 2 | 3.3917 | 6.5865 | 1.0060 | 8.0000 |
| 830010 | overpass | 1 / 0 / 2 | 2 | 4.0127 | 5.2807 | 0.4572 | None |
| 830011 | narrow_crossing | 2 / 0 / 1 | 1 | 4.7763 | 7.1402 | 0.3155 | None |
| 830012 | dynamic_crossing | 2 / 0 / 1 | 1 | 5.6664 | 7.0532 | 0.4691 | None |

## treatment: map descriptions

| Map seed | Family | Never / ball-only / captured agents | Max occupancy | Worst closest m | Worst final m | Worst final speed m/s | First-capture gap s |
|---|---|---|---:|---:|---:|---:|---|
| 830001 | urban_canyon | 0 / 0 / 3 | 3 | 3.9422 | 3.9422 | 1.2300 | 1.6000 |
| 830002 | warehouse | 1 / 1 / 1 | 1 | 5.3569 | 5.4463 | 0.1421 | None |
| 830003 | courtyard | 0 / 1 / 2 | 2 | 3.6635 | 5.3850 | 0.3938 | None |
| 830004 | overpass | 0 / 0 / 3 | 3 | 3.9959 | 3.9959 | 0.5629 | 6.4000 |
| 830005 | narrow_crossing | 1 / 0 / 2 | 2 | 4.4292 | 11.3757 | 0.1365 | None |
| 830006 | dynamic_crossing | 1 / 0 / 2 | 2 | 4.2469 | 4.4172 | 0.3089 | None |
| 830007 | urban_canyon | 0 / 0 / 3 | 3 | 3.9986 | 3.9986 | 0.5714 | 7.8000 |
| 830008 | warehouse | 1 / 1 / 1 | 1 | 4.4997 | 5.2353 | 0.1073 | None |
| 830009 | courtyard | 0 / 0 / 3 | 2 | 2.7806 | 4.6120 | 0.3668 | 10.2000 |
| 830010 | overpass | 0 / 0 / 3 | 2 | 3.8473 | 4.7908 | 0.2951 | 1.6000 |
| 830011 | narrow_crossing | 0 / 0 / 3 | 3 | 3.8605 | 3.8605 | 1.3836 | 6.8000 |
| 830012 | dynamic_crossing | 0 / 0 / 3 | 3 | 3.9715 | 3.9715 | 1.0841 | 6.0000 |

Per-agent minimum/final distance, final speed, capture-frame count, longest capture streak and first-capture time are retained in summary.json. All requested JSON/trajectory cross-checks passed; inconsistencies raise an error instead of being hidden.
