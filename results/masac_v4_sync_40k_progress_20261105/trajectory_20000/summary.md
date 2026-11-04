# Post-hoc trajectory diagnosis: seed 1511, step 20000

Post-hoc intermediate diagnosis only. The registered final40k primary endpoint and six-history advancement rule remain unchanged; no training change, automatic extension or sealed-test evaluation is authorized.

Each arm has 36 correlated agent-map observations from one training seed and 12 development maps; these are descriptive counts, not 36 independent trials.

Capture occupancy counts position/speed only. Success remains collision-free joint capture; all four nonexclusive collision flags are retained per map and per arm in summary.json.

min_distance_m and JSON closest include reset frame, matching evaluate(); capture/ever-ball/streak/timing use only post-step frames.

Range of first-capture times only if every agent ever captures; null otherwise. Eventual capture and first-arrival gap do not establish simultaneous capture.

| Arm | Never ball | Ball without capture | Ever capture | Success maps | Any-ball maps | All-ball maps | All-ever-capture maps |
|---|---:|---:|---:|---:|---:|---:|---:|
| control | 27 | 4 | 5 | 0/12 | 8/12 | 0/12 | 0/12 |
| treatment | 22 | 2 | 12 | 0/12 | 8/12 | 2/12 | 2/12 |

## control: map descriptions

| Map seed | Family | Never / ball-only / captured agents | Max occupancy | Worst closest m | Worst final m | Worst final speed m/s | First-capture gap s |
|---|---|---|---:|---:|---:|---:|---|
| 830001 | urban_canyon | 3 / 0 / 0 | 0 | 8.7649 | 27.0860 | 6.4434 | None |
| 830002 | warehouse | 2 / 0 / 1 | 1 | 5.0853 | 17.2563 | 5.3746 | None |
| 830003 | courtyard | 2 / 1 / 0 | 0 | 8.2831 | 25.4819 | 5.8818 | None |
| 830004 | overpass | 2 / 0 / 1 | 1 | 14.3693 | 24.2442 | 6.7894 | None |
| 830005 | narrow_crossing | 3 / 0 / 0 | 0 | 17.5491 | 27.0781 | 3.6155 | None |
| 830006 | dynamic_crossing | 1 / 2 / 0 | 0 | 5.6919 | 16.7756 | 5.1945 | None |
| 830007 | urban_canyon | 3 / 0 / 0 | 0 | 7.9972 | 20.1199 | 6.4622 | None |
| 830008 | warehouse | 2 / 0 / 1 | 1 | 6.0266 | 11.6201 | 4.4731 | None |
| 830009 | courtyard | 2 / 0 / 1 | 1 | 7.3634 | 16.2553 | 4.5950 | None |
| 830010 | overpass | 3 / 0 / 0 | 0 | 13.1954 | 25.0933 | 5.1359 | None |
| 830011 | narrow_crossing | 2 / 1 / 0 | 0 | 6.8365 | 15.2728 | 4.9496 | None |
| 830012 | dynamic_crossing | 2 / 0 / 1 | 1 | 10.5585 | 22.8669 | 6.3324 | None |

## treatment: map descriptions

| Map seed | Family | Never / ball-only / captured agents | Max occupancy | Worst closest m | Worst final m | Worst final speed m/s | First-capture gap s |
|---|---|---|---:|---:|---:|---:|---|
| 830001 | urban_canyon | 0 / 0 / 3 | 1 | 3.7646 | 15.1050 | 0.2251 | 10.6000 |
| 830002 | warehouse | 2 / 0 / 1 | 1 | 6.9204 | 22.0831 | 0.0747 | None |
| 830003 | courtyard | 2 / 0 / 1 | 1 | 7.3359 | 22.1386 | 0.1187 | None |
| 830004 | overpass | 3 / 0 / 0 | 0 | 8.5570 | 14.7529 | 0.1108 | None |
| 830005 | narrow_crossing | 3 / 0 / 0 | 0 | 8.9518 | 30.1728 | 0.2551 | None |
| 830006 | dynamic_crossing | 2 / 1 / 0 | 0 | 4.9114 | 18.6110 | 0.1694 | None |
| 830007 | urban_canyon | 1 / 0 / 2 | 1 | 4.7894 | 16.4801 | 0.1539 | None |
| 830008 | warehouse | 3 / 0 / 0 | 0 | 8.5762 | 24.9413 | 0.1042 | None |
| 830009 | courtyard | 1 / 0 / 2 | 1 | 4.5785 | 22.1996 | 0.1667 | None |
| 830010 | overpass | 3 / 0 / 0 | 0 | 8.0242 | 13.3650 | 0.1148 | None |
| 830011 | narrow_crossing | 2 / 1 / 0 | 0 | 10.9163 | 36.5976 | 0.2154 | None |
| 830012 | dynamic_crossing | 0 / 0 / 3 | 1 | 2.8767 | 13.1983 | 0.1762 | 14.0000 |

Per-agent minimum/final distance, final speed, capture-frame count, longest capture streak and first-capture time are retained in summary.json. All requested JSON/trajectory cross-checks passed; inconsistencies raise an error instead of being hidden.
