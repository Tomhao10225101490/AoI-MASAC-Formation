# Post-hoc trajectory diagnosis: seed 1511, step 40000

Post-hoc intermediate diagnosis only. The registered final40k primary endpoint and six-history advancement rule remain unchanged; no training change, automatic extension or sealed-test evaluation is authorized.

Each arm has 36 correlated agent-map observations from one training seed and 12 development maps; these are descriptive counts, not 36 independent trials.

Capture occupancy counts position/speed only. Success remains collision-free joint capture; all four nonexclusive collision flags are retained per map and per arm in summary.json.

min_distance_m and JSON closest include reset frame, matching evaluate(); capture/ever-ball/streak/timing use only post-step frames.

Range of first-capture times only if every agent ever captures; null otherwise. Eventual capture and first-arrival gap do not establish simultaneous capture.

| Arm | Never ball | Ball without capture | Ever capture | Success maps | Any-ball maps | All-ball maps | All-ever-capture maps |
|---|---:|---:|---:|---:|---:|---:|---:|
| control | 7 | 1 | 28 | 2/12 | 12/12 | 5/12 | 5/12 |
| treatment | 6 | 0 | 30 | 6/12 | 12/12 | 8/12 | 8/12 |

## control: map descriptions

| Map seed | Family | Never / ball-only / captured agents | Max occupancy | Worst closest m | Worst final m | Worst final speed m/s | First-capture gap s |
|---|---|---|---:|---:|---:|---:|---|
| 830001 | urban_canyon | 0 / 0 / 3 | 2 | 3.8357 | 4.8913 | 0.6219 | 16.8000 |
| 830002 | warehouse | 1 / 0 / 2 | 2 | 4.1235 | 9.3204 | 0.3791 | None |
| 830003 | courtyard | 1 / 0 / 2 | 1 | 5.8378 | 7.5866 | 0.4007 | None |
| 830004 | overpass | 1 / 1 / 1 | 1 | 4.0071 | 11.8351 | 0.3475 | None |
| 830005 | narrow_crossing | 1 / 0 / 2 | 2 | 7.0505 | 8.8798 | 0.4006 | None |
| 830006 | dynamic_crossing | 0 / 0 / 3 | 2 | 3.2733 | 6.0939 | 0.3286 | 4.6000 |
| 830007 | urban_canyon | 0 / 0 / 3 | 3 | 3.8964 | 3.8964 | 1.4783 | 3.8000 |
| 830008 | warehouse | 0 / 0 / 3 | 3 | 3.8016 | 3.8381 | 1.7534 | 1.0000 |
| 830009 | courtyard | 1 / 0 / 2 | 2 | 5.4651 | 10.7061 | 0.3283 | None |
| 830010 | overpass | 1 / 0 / 2 | 2 | 5.6294 | 8.2540 | 0.3516 | None |
| 830011 | narrow_crossing | 1 / 0 / 2 | 2 | 4.1793 | 6.0567 | 0.3824 | None |
| 830012 | dynamic_crossing | 0 / 0 / 3 | 2 | 3.3547 | 6.5642 | 0.2877 | 13.6000 |

## treatment: map descriptions

| Map seed | Family | Never / ball-only / captured agents | Max occupancy | Worst closest m | Worst final m | Worst final speed m/s | First-capture gap s |
|---|---|---|---:|---:|---:|---:|---|
| 830001 | urban_canyon | 0 / 0 / 3 | 3 | 3.9912 | 3.9912 | 0.5512 | 7.6000 |
| 830002 | warehouse | 2 / 0 / 1 | 1 | 5.6006 | 6.5387 | 0.9653 | None |
| 830003 | courtyard | 0 / 0 / 3 | 3 | 3.9475 | 3.9475 | 1.2992 | 5.2000 |
| 830004 | overpass | 0 / 0 / 3 | 3 | 3.6335 | 3.6335 | 1.7954 | 4.0000 |
| 830005 | narrow_crossing | 0 / 0 / 3 | 3 | 3.8394 | 3.9953 | 0.4234 | 11.2000 |
| 830006 | dynamic_crossing | 1 / 0 / 2 | 1 | 4.3978 | 8.4194 | 0.3288 | None |
| 830007 | urban_canyon | 1 / 0 / 2 | 2 | 5.0508 | 6.3326 | 0.5864 | None |
| 830008 | warehouse | 2 / 0 / 1 | 1 | 7.7992 | 8.8272 | 0.5895 | None |
| 830009 | courtyard | 0 / 0 / 3 | 2 | 3.9278 | 4.9901 | 0.3460 | 12.6000 |
| 830010 | overpass | 0 / 0 / 3 | 3 | 3.8080 | 3.8080 | 1.5986 | 2.2000 |
| 830011 | narrow_crossing | 0 / 0 / 3 | 3 | 3.9564 | 3.9564 | 0.3746 | 9.0000 |
| 830012 | dynamic_crossing | 0 / 0 / 3 | 2 | 2.5384 | 6.9631 | 0.7177 | 3.6000 |

Per-agent minimum/final distance, final speed, capture-frame count, longest capture streak and first-capture time are retained in summary.json. All requested JSON/trajectory cross-checks passed; inconsistencies raise an error instead of being hidden.
