# Offline near-goal actor response

Exactly Phase A: six static/control+B late-training replay samples, 1024 joint observations per history. Not final-policy on-policy states, historical full-policy states, or sealed data.

Posthoc descriptive actor-response diagnosis. Pure one-step kinematic counterfactuals are not collision feasibility, episode success, policy optimality, causality, efficacy, or an advancement decision.

Deterministic response is tanh(mu), not the expectation of a squashed Gaussian. The 64 common 3D noise vectors provide per-agent marginal MC summaries only; there is no team joint-policy MC probability.

One-step kinematic strict capture is distance<4m and speed<2m/s after the exact frozen limiter. It omits collisions and future interactions. Starting distance>=4.4m makes strict single-step capture impossible at dt=.2; that necessary bound is not a feasibility certificate.

| Arm | Seed | Subset | Agent observations | Deterministic strict capture | MC marginal strict capture / actions | Mean inward acceleration (det) | Mean distance change (det) | Paired MC−det distance change |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| control | 1511 | all | 3072 | 287/3072 | 18334/196608 | 2.04819 | -0.170996 | 0.00308497 |
| control | 1511 | primary_d4_10_speed_lt2 | 1115 | 9/1115 | 635/71360 | 1.78907 | -0.0662205 | 0.00378269 |
| control | 1511 | boundary_d4_4p4_speed_lt2 | 83 | 9/83 | 635/5312 | 1.3939 | -0.0632384 | 0.00115298 |
| control | 1522 | all | 3072 | 603/3072 | 38542/196608 | 1.2971 | -0.1421 | 0.00136805 |
| control | 1522 | primary_d4_10_speed_lt2 | 1364 | 18/1364 | 1164/87296 | 1.38974 | -0.0493209 | 0.0014256 |
| control | 1522 | boundary_d4_4p4_speed_lt2 | 101 | 18/101 | 1164/6464 | 1.39717 | -0.0674876 | 0.000810595 |
| control | 1533 | all | 3072 | 773/3072 | 49061/196608 | 1.67831 | -0.189995 | 0.00219176 |
| control | 1533 | primary_d4_10_speed_lt2 | 1418 | 43/1418 | 2553/90752 | 1.91328 | -0.0673247 | 0.00239193 |
| control | 1533 | boundary_d4_4p4_speed_lt2 | 172 | 43/172 | 2553/11008 | 1.61872 | -0.0682392 | 0.00299617 |
| b | 1511 | all | 3072 | 628/3072 | 40074/196608 | 1.82217 | -0.171114 | 0.00247066 |
| b | 1511 | primary_d4_10_speed_lt2 | 1563 | 34/1563 | 2054/100032 | 1.97158 | -0.0741242 | 0.00252377 |
| b | 1511 | boundary_d4_4p4_speed_lt2 | 183 | 34/183 | 2054/11712 | 1.87602 | -0.0645378 | 0.00279688 |
| b | 1522 | all | 3072 | 412/3072 | 26249/196608 | 1.82337 | -0.15594 | 0.00206455 |
| b | 1522 | primary_d4_10_speed_lt2 | 1101 | 18/1101 | 1227/70464 | 2.11768 | -0.0739808 | 0.00266484 |
| b | 1522 | boundary_d4_4p4_speed_lt2 | 81 | 18/81 | 1227/5184 | 1.5141 | -0.0539913 | 0.00134649 |
| b | 1533 | all | 3072 | 847/3072 | 53774/196608 | 1.77983 | -0.204731 | 0.0022639 |
| b | 1533 | primary_d4_10_speed_lt2 | 1439 | 46/1439 | 2807/92096 | 2.24075 | -0.0947317 | 0.00278658 |
| b | 1533 | boundary_d4_4p4_speed_lt2 | 182 | 46/182 | 2807/11648 | 2.11561 | -0.0870368 | 0.00275329 |
| control | pooled | all | 9216 | 1663/9216 | 105937/589824 | 1.67453 | -0.167697 | 0.00221493 |
| control | pooled | primary_d4_10_speed_lt2 | 3897 | 70/3897 | 4352/249408 | 1.6945 | -0.0607072 | 0.00245162 |
| control | pooled | boundary_d4_4p4_speed_lt2 | 356 | 70/356 | 4352/22784 | 1.50345 | -0.06686 | 0.00194637 |
| b | pooled | all | 9216 | 1887/9216 | 120097/589824 | 1.80846 | -0.177262 | 0.00226637 |
| b | pooled | primary_d4_10_speed_lt2 | 4103 | 98/4103 | 6088/262592 | 2.10519 | -0.0813131 | 0.00265379 |
| b | pooled | boundary_d4_4p4_speed_lt2 | 446 | 98/446 | 6088/28544 | 1.90806 | -0.0718036 | 0.00251568 |
| all | pooled | all | 18432 | 3550/18432 | 226034/1179648 | 1.74149 | -0.172479 | 0.00224065 |
| all | pooled | primary_d4_10_speed_lt2 | 8000 | 168/8000 | 10440/512000 | 1.90513 | -0.0712755 | 0.00255531 |
| all | pooled | boundary_d4_4p4_speed_lt2 | 802 | 168/802 | 10440/51328 | 1.72846 | -0.0696092 | 0.00226297 |

All distance/speed bins, parameter moments, radial/tangential statistics, threshold-proximity counts, per-state paired MC-minus-deterministic summaries and empty-group denominators are in audit.json and groups.csv. Population standard deviations describe correlated observations, not uncertainty over independent seeds.

Float64 geometry starts from quantized float32 observations; 1e-4 threshold-proximity counts do not exclude observations or relax strict capture. Zero goal distance uses a zero inward vector, and undefined radial cosines use zero with explicit counts. No nonfinite values were replaced or omitted.

Each actor used only its own SHA-verified checkpoint. Packed critic/optimizer objects may be deserialized as part of the trusted local container but are not consumed; no learner, environment, update, Q evaluation, map rollout, sealed evaluation, or saved RNG restoration occurs. Actor values and global torch RNG states are checked unchanged.

No causal, efficacy, optimality, final-policy-occupancy or advancement conclusion is produced. Cross-arm replay states are different populations; they are not paired same-state interventions.
