# Post-hoc trajectory diagnosis: seed 1511, step 22000

Post-hoc intermediate diagnosis only. The registered final40k primary endpoint and six-history advancement rule remain unchanged; no training change, automatic extension or sealed-test evaluation is authorized.

Each arm has 36 correlated agent-map observations from one training seed and 12 development maps; these are descriptive counts, not 36 independent trials.

Capture occupancy counts position/speed only. Success remains collision-free joint capture; all four nonexclusive collision flags are retained per map and per arm in summary.json.

min_distance_m and JSON closest include reset frame, matching evaluate(); capture/ever-ball/streak/timing use only post-step frames.

Range of first-capture times only if every agent ever captures; null otherwise. Eventual capture and first-arrival gap do not establish simultaneous capture.

| Arm | Never ball | Ball without capture | Ever capture | Success maps | Any-ball maps | All-ball maps | All-ever-capture maps |
|---|---:|---:|---:|---:|---:|---:|---:|
| control | 20 | 9 | 7 | 0/12 | 11/12 | 0/12 | 0/12 |
| treatment | 24 | 0 | 12 | 0/12 | 10/12 | 0/12 | 0/12 |

## control: map descriptions

| Map seed | Family | Never / ball-only / captured agents | Max occupancy | Worst closest m | Worst final m | Worst final speed m/s | First-capture gap s |
|---|---|---|---:|---:|---:|---:|---|
| 830001 | urban_canyon | 2 / 1 / 0 | 0 | 4.6882 | 28.6109 | 5.6154 | None |
| 830002 | warehouse | 1 / 0 / 2 | 1 | 16.6378 | 37.9746 | 4.5069 | None |
| 830003 | courtyard | 2 / 1 / 0 | 0 | 7.4000 | 15.3168 | 4.5525 | None |
| 830004 | overpass | 3 / 0 / 0 | 0 | 12.7115 | 40.7312 | 5.3111 | None |
| 830005 | narrow_crossing | 1 / 2 / 0 | 0 | 4.2142 | 32.3002 | 3.1065 | None |
| 830006 | dynamic_crossing | 2 / 1 / 0 | 0 | 15.7195 | 27.0676 | 2.9059 | None |
| 830007 | urban_canyon | 2 / 0 / 1 | 1 | 6.1061 | 33.5246 | 3.7406 | None |
| 830008 | warehouse | 1 / 1 / 1 | 1 | 10.8265 | 36.0454 | 7.7434 | None |
| 830009 | courtyard | 2 / 0 / 1 | 1 | 6.8840 | 28.4909 | 6.0161 | None |
| 830010 | overpass | 2 / 1 / 0 | 0 | 12.2441 | 50.5967 | 7.1628 | None |
| 830011 | narrow_crossing | 1 / 0 / 2 | 1 | 11.1410 | 49.5781 | 8.1132 | None |
| 830012 | dynamic_crossing | 1 / 2 / 0 | 0 | 4.0201 | 27.5589 | 3.8664 | None |

## treatment: map descriptions

| Map seed | Family | Never / ball-only / captured agents | Max occupancy | Worst closest m | Worst final m | Worst final speed m/s | First-capture gap s |
|---|---|---|---:|---:|---:|---:|---|
| 830001 | urban_canyon | 2 / 0 / 1 | 1 | 7.3622 | 11.3562 | 0.0573 | None |
| 830002 | warehouse | 2 / 0 / 1 | 1 | 6.0282 | 11.7880 | 0.0125 | None |
| 830003 | courtyard | 3 / 0 / 0 | 0 | 6.8605 | 11.3332 | 0.2741 | None |
| 830004 | overpass | 1 / 0 / 2 | 2 | 6.8180 | 9.1352 | 0.5263 | None |
| 830005 | narrow_crossing | 1 / 0 / 2 | 1 | 4.0977 | 8.0342 | 0.0307 | None |
| 830006 | dynamic_crossing | 2 / 0 / 1 | 1 | 6.2837 | 9.0951 | 0.0570 | None |
| 830007 | urban_canyon | 2 / 0 / 1 | 1 | 7.3090 | 11.5390 | 0.0242 | None |
| 830008 | warehouse | 2 / 0 / 1 | 1 | 7.0122 | 9.4003 | 0.0221 | None |
| 830009 | courtyard | 2 / 0 / 1 | 1 | 7.8025 | 8.2371 | 0.0035 | None |
| 830010 | overpass | 3 / 0 / 0 | 0 | 8.1014 | 9.7397 | 0.7661 | None |
| 830011 | narrow_crossing | 2 / 0 / 1 | 1 | 10.1452 | 16.6447 | 0.0047 | None |
| 830012 | dynamic_crossing | 2 / 0 / 1 | 1 | 6.0913 | 10.6668 | 0.0128 | None |

Per-agent minimum/final distance, final speed, capture-frame count, longest capture streak and first-capture time are retained in summary.json. All requested JSON/trajectory cross-checks passed; inconsistencies raise an error instead of being hidden.
