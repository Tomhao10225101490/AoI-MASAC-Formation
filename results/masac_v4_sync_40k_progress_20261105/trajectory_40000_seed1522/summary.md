# Post-hoc trajectory diagnosis: seed 1522, step 40000

Post-hoc intermediate diagnosis only. The registered final40k primary endpoint and six-history advancement rule remain unchanged; no training change, automatic extension or sealed-test evaluation is authorized.

Each arm has 36 correlated agent-map observations from one training seed and 12 development maps; these are descriptive counts, not 36 independent trials.

Capture occupancy counts position/speed only. Success remains collision-free joint capture; all four nonexclusive collision flags are retained per map and per arm in summary.json.

min_distance_m and JSON closest include reset frame, matching evaluate(); capture/ever-ball/streak/timing use only post-step frames.

Range of first-capture times only if every agent ever captures; null otherwise. Eventual capture and first-arrival gap do not establish simultaneous capture.

| Arm | Never ball | Ball without capture | Ever capture | Success maps | Any-ball maps | All-ball maps | All-ever-capture maps |
|---|---:|---:|---:|---:|---:|---:|---:|
| control | 20 | 1 | 15 | 1/12 | 9/12 | 1/12 | 1/12 |
| treatment | 17 | 0 | 19 | 0/12 | 11/12 | 2/12 | 2/12 |

## control: map descriptions

| Map seed | Family | Never / ball-only / captured agents | Max occupancy | Worst closest m | Worst final m | Worst final speed m/s | First-capture gap s |
|---|---|---|---:|---:|---:|---:|---|
| 830001 | urban_canyon | 1 / 1 / 1 | 1 | 5.0354 | 13.0629 | 0.0655 | None |
| 830002 | warehouse | 3 / 0 / 0 | 0 | 7.0477 | 8.9364 | 0.7099 | None |
| 830003 | courtyard | 2 / 0 / 1 | 1 | 5.6165 | 12.6923 | 0.1007 | None |
| 830004 | overpass | 1 / 0 / 2 | 2 | 9.5157 | 11.6429 | 0.0003 | None |
| 830005 | narrow_crossing | 2 / 0 / 1 | 1 | 5.4582 | 9.3510 | 0.4815 | None |
| 830006 | dynamic_crossing | 1 / 0 / 2 | 2 | 4.5206 | 10.0530 | 0.3271 | None |
| 830007 | urban_canyon | 1 / 0 / 2 | 2 | 4.5429 | 7.3139 | 0.1820 | None |
| 830008 | warehouse | 2 / 0 / 1 | 1 | 5.9338 | 7.3324 | 0.9945 | None |
| 830009 | courtyard | 0 / 0 / 3 | 3 | 3.0501 | 3.9980 | 1.3697 | 0.8000 |
| 830010 | overpass | 3 / 0 / 0 | 0 | 8.6578 | 10.2091 | 0.0106 | None |
| 830011 | narrow_crossing | 1 / 0 / 2 | 2 | 4.6647 | 5.4307 | 0.2991 | None |
| 830012 | dynamic_crossing | 3 / 0 / 0 | 0 | 8.2427 | 9.6337 | 0.0049 | None |

## treatment: map descriptions

| Map seed | Family | Never / ball-only / captured agents | Max occupancy | Worst closest m | Worst final m | Worst final speed m/s | First-capture gap s |
|---|---|---|---:|---:|---:|---:|---|
| 830001 | urban_canyon | 0 / 0 / 3 | 2 | 3.9130 | 7.0663 | 2.2220 | 6.0000 |
| 830002 | warehouse | 2 / 0 / 1 | 1 | 6.3326 | 7.8852 | 0.3473 | None |
| 830003 | courtyard | 0 / 0 / 3 | 2 | 3.5128 | 5.1217 | 1.1752 | 10.0000 |
| 830004 | overpass | 2 / 0 / 1 | 1 | 7.2596 | 11.3405 | 0.6070 | None |
| 830005 | narrow_crossing | 1 / 0 / 2 | 1 | 6.5111 | 9.0821 | 0.7082 | None |
| 830006 | dynamic_crossing | 3 / 0 / 0 | 0 | 6.1502 | 9.7207 | 1.7790 | None |
| 830007 | urban_canyon | 2 / 0 / 1 | 1 | 5.4907 | 9.3373 | 0.6533 | None |
| 830008 | warehouse | 1 / 0 / 2 | 2 | 7.1261 | 7.7596 | 1.0227 | None |
| 830009 | courtyard | 2 / 0 / 1 | 1 | 5.1866 | 8.0441 | 0.8808 | None |
| 830010 | overpass | 2 / 0 / 1 | 1 | 7.9597 | 11.1599 | 0.3242 | None |
| 830011 | narrow_crossing | 1 / 0 / 2 | 1 | 4.6655 | 8.8790 | 0.8001 | None |
| 830012 | dynamic_crossing | 1 / 0 / 2 | 1 | 4.9615 | 8.2831 | 0.6357 | None |

Per-agent minimum/final distance, final speed, capture-frame count, longest capture streak and first-capture time are retained in summary.json. All requested JSON/trajectory cross-checks passed; inconsistencies raise an error instead of being hidden.
