# PAPER_PLAN_WOWMOM2027 — Predictive Maneuverability-Aware MASAC

**2026-11-05 evidence update:** All12 archived L0 development maps have zero
external moving objects. Full/static cannot identify moving-obstacle prediction;
new early-replay evidence also exposes neutral-value saturation confounds
(`L0_CONTENT_IDENTIFIABILITY_AUDIT_20261105.md`). The two-seed2k B/control pilot
is complete with0/12 final success in all four runs and finite diagnostics.
The separate fixed 40k × three-seed static B-off/on study is COMPLETE. Control
finals are 2/12, 1/12, 4/12; B finals are 6/12, 0/12, 5/12. The original
advancement rule is met for planning confirmation only; both absolute gates
remain FAIL 0/3. It tests known potential shaping, not predictive-content
novelty or a full curriculum. See
`design/MASAC_V4_SYNC_CAPTURE_40K_PROTOCOL_20261105.md`.

**Created:** 2026-11-04 15:20 Asia/Shanghai  
**Repo:** https://github.com/Tomhao10225101490/masac-maneuverability  
**Working branch:** `research/wowmom2027-upgrade` (keep `main` in sync)  
**Updated:** 2026-11-05 Asia/Shanghai. The historical Mac 80k suite and the separate Windows 40k control/B study are COMPLETE. No arm passes its absolute all-seed task gate. The 40k engineering advancement rule permits planning confirmation only. No significance, moving-obstacle benefit, safety, convergence or novelty conclusion is established; new large launches require a visible plan first.

---

## 0. One-sentence story

We study observation-only fixed-control preview under strict simultaneous goal settling, with two separate experiments: the historical Mac 80k full/static/zero content-mask suite and the Windows 40k static control/B fixed-potential study. Static has the strongest historical mean; the new B mean exceeds its paired control by 11.111 percentage points, with one negative seed and all absolute gates failed. L0 has no external movers, so predictive-content utility remains unidentified. Legacy Standard-C is a separate warm-started learnability floor, not a fair performance comparator.

---

## 1. Problem & framing (WoWMoM-facing)

**Task.** \(N=3\) UAVs, continuous 3D acceleration actions, independent goals on frozen L0 city maps; success = **joint** capture (all agents in ball with speed limit, latch off).

**Measured bottleneck.** Historical `ever_ball_maps` means **at least one** UAV ever entered its ball. Of the 65 final suite failures, 61 have only some UAVs ever entering; four have all UAVs entering at some times without joint success; none collide. Thus worst-agent goal approach and joint capture are the immediate targets. These counts do not identify a purely synchronization-caused failure or establish which mechanism will fix it.

**Why WoWMoM / mobile AI systems.** Fits AI/ML for mobile cyber-physical agents, navigation/mobility management, and reproducible open artifacts — **not** a radio/AoI paper unless a later extension adds communication freshness. Keep AoI only as adjacent related work.

**Venue notes (IEEE WoWMoM 2027).**
- Event: Irvine, CA, **21–24 June 2027** ([CFP](https://wowmom2027.dii.unipi.it/call-for-papers.html)).
- Official regular-paper deadline: **2026-12-01 23:59 AoE**, equivalent to **2026-12-02 19:59 Asia/Shanghai**.
- Official regular-paper rules: **10 pages including figures/references**, font **10 pt or larger**, IEEE two-column, **double blind**, in-person presentation. See [official CFP](https://wowmom2027.dii.unipi.it/call-for-papers.html) and `FRONTIER_GAP_UPDATE_20261105.md`; recheck at submission.
- Align narrative to: AI-assisted mobile systems, localization/navigation/mobility, reproducible artifacts — **de-emphasize** pure control-theory novelty if venue reviewers expect networking/systems angle.
- Fallback venues if WoWMoM fit is weak after results: ICRA / IROS / Drones journal / IEEE T-ITS short — decide after suite closes.

---

## 2. Contributions (draft C1–C3 — claim only after evidence)

| ID | Draft claim | Evidence needed before claiming |
|----|-------------|----------------------------------|
| **C1** | An executable observation-only control-bank representation, precisely defined by E5–E6. | Source/geometry correspondence. Feature construction is implemented; distinctiveness from preview/risk prior art remains to establish. |
| **C2** | A controlled within-suite content ablation with mixed/negative evidence for adding CV moving/peer channels. | Final paired-seed `static` / `full` / `zero` results below; mean ranking is descriptive and reverses static/full on seed 1233. |
| **C3** | A reproducible fixed-endpoint study and temporal-event failure taxonomy, with a fixed-potential engineering response. | Verified 40k final artifacts; original advancement rule met for planning confirmation only, absolute gate FAIL 0/3 in both arms. Distinguish ball entry, individually attained strict capture and simultaneous capture; Standard-C remains a separate floor. |

**Non-claims (explicit).**
- Probes are **not** continuous-time certificates, not joint multi-agent safety filters, not differential-flatness planners.
- Path-C reward reshape is **engineering curriculum**, not scientific novelty.
- Do **not** claim parameter-matched vs Legacy Standard-C encoder.
- Do **not** claim Attention / L1 / communication until separately gated.
- Static still previews ego motion for two seconds; full-vs-static is not prediction-vs-no-prediction.
- Zero removes direction coordinates as well as risk scalars; its trainable encoder can learn a constant. Its failure does not prove that capacity is useless or that a learned actor causally uses the risk embedding.
- B fixed potential shaping has completed descriptive evidence, including a negative seed. Its standard identity and engineering advancement rule establish neither significance nor a confirmed learning/capture advantage.

---

## 3. Separate experimental cohorts

### 3.1 Legacy Standard-C learnability floor

Seeds 911/922/933 finish at **11/12, 11/12, 10/12**, PASS 3/3, after 50k L0 steps. Each loads the same empty-floor seed-811 40k checkpoint with weights and optimizers, resetting replay/RNG. Architecture, initialization history, seeds, budget and machine differ from the later studies. This is a separate task learnability floor; it is not a fair performance comparator for either cold-start cohort.

### 3.2 Historical Mac 80k content-mask suite

Within-suite contract (**full/static/zero only**): C reshape; capture **4 m / 2 m·s⁻¹**, `latch=False`; cold start; **80k×3** seeds **1211/1222/1233**; gate ≥0.60 (8/12); L0 maps `830001`–`830012`.

| Historical 80k arm | Role | Final result (suite completed 2026-11-04) |
|----------------|------|----------------------------------------|
| **maneuver-zero** | Matched-parameter constant-risk-input control | **FAIL 0/3**: 1/12, 0/12, 1/12; best ~5/12 |
| **maneuver-static** | Static/boundary geometry preview; excludes mv+peer from derived aggregates | **FAIL 2/3**: 11/12, 10/12, 5/12; seed 1233 falls from 9/12 at 70k to 5/12 |
| **maneuver-full** | Full CV probes (candidate) | **FAIL 1/3**: 3/12, 4/12, 8/12; seed 1233 alone passes |
| shuffle (optional) | Direction–feature permutation control | Not launched |

**Evidence boundary:** Standard-C loads the same empty-floor seed-811 40k checkpoint for each L0 seed, retaining weights and optimizers while resetting replay/RNG. Different architecture, initialization history, seeds and 50k L0 budget confound comparison with the 80k cold-start maneuver arms.

Descriptive final totals are static **26/36**, full **15/36**, zero **2/36**. These are three training seeds evaluated on the same 12 development maps, not 36 independent training trials. No arm passes all three seeds. Preserve finals and learning curves; do not replace seed 1233's final static value by its best checkpoint. No sealed unseen-map generalization claim is available.

---

### 3.3 Windows 40k static control/B study — COMPLETE

The new experiment uses cold seeds **1511/1522/1533**, static probes, fixed B coefficient **0.5**, temperature **0.1**, unchanged evaluation and exactly **40k** steps per seed. It is not an extension or rerun of the historical 80k suite. All six histories and final artifacts are verified; source/configuration and decision rules were not changed.

| 40k arm | 1511 | 1522 | 1533 | Absolute 8/12-per-seed gate |
|---|---|---|---|---|
| static control | 2/12 | 1/12 | 4/12 | FAIL 0/3 |
| static+B | 6/12 | 0/12 | 5/12 | FAIL 0/3 |
| Paired success-count difference | +4 | −1 | +1 | Not a task-gate replacement |

The descriptive totals are control **7/36** and B **11/36**; the mean paired success-fraction difference is **1/9 = 11.111 percentage points**. The original rule is met: mean difference at least 1/12, at least two positive pairs, no pair below −1/12, and all six histories finite with the contracts unchanged. Seed 1522 lies exactly on the allowed negative bound. **ADVANCEMENT_RULE_MET permits planning confirmation only.** It is not statistical significance, an absolute task pass, a dynamic-prediction benefit or authorization for another launch.

Use the [canonical final analysis](../results/masac_v4_sync_40k_analysis_20261105/analysis.md). Three training-seed pairs and repeatedly observed development maps support a descriptive result only; the 12 reserved test maps remain sealed.

### 3.4 Final 40k settling-event diagnosis

Let A mean all UAVs ever enter their balls, Cevt mean all UAVs individually attain strict ball-plus-speed capture at some times, and J mean simultaneous strict capture. The temporal nesting is J implies Cevt implies A. These definitions add no reward or rule change.

| Seed | Control (A, Cevt, J), out of 12 | B (A, Cevt, J), out of 12 |
|---|---|---|
| 1511 | (5, 5, 2) | (8, 8, 6) |
| 1522 | (1, 1, 1) | (2, 2, 0) |
| 1533 | (5, 5, 4) | (8, 7, 5) |

Across 72 final rollouts there are **18 successes and 54 failures**, with **zero collisions**. The exhaustive collision-free failure partition is **43 not-all-ball-entry**, **one all-ball-entry without all-individual-strict-capture**, and **10 all-individual-strict-capture without joint capture**. These are observed event classes, not identified causes. Do not pool this 40k diagnosis with the Mac 80k suite's 65 failures or infer that B caused a change in synchronization difficulty from raw category totals.

### 3.5 Late-replay input audit — a bounded population, not learning equivalence

The [new late-replay audit](../results/masac_v4_sync_40k_late_content_audit_20261105/audit.md) covers **6,144 joint observations** from the six completed 40k histories: **18,432 agent observations and 350,208 probe vectors**. Full equals static_fill1 in **349,849/350,208 vectors (99.89748949%)**; all **359** exceptions differ only in the peer column and occur in control/1522. The other nine columns, including TTC/brake/reserve/flag, are bitwise identical between full/static; moving outputs are always full=1, static=0. Every audited agent observation has two valid peers: saturated peer=1 mostly reflects valid far peers plus the 40 m padding sentinel, not absent peers. Five of six histories have complete vector identity within this audited sample.

These are same-observation feature calculations on the declared sampled population, not the entire replay, historical Mac 80k suite, a policy rollout or sealed-map evidence. They do not establish equal learned functions/optimization trajectories or identify constant fill as a performance cause. The original experiment and rule remain unchanged.

## 4. Evidence and next work before stronger claims

1. Freeze the completed three-seed finals above; carry seed variability and the static 1233 drop into the abstract and results.
2. Report joint success, **any** entry, **all eventual** entry, collision and success-conditional path efficiency separately. All-eventual entry counts: static **11,10,5**; full **5,4,8**; zero **1,1,2**.
3. Use the mixed/negative story: static-geometry preview is strongest on average, but the value of CV moving/peer channels remains unestablished. Correlations and final ranks do not identify a failure cause.
4. B implementation, the 2k sanity pilot and the registered 40k six-run study are complete. Report its fixed endpoints and negative seed; its original rule permits planning confirmation only. Prepare an offline constant-fill/input-content diagnosis before proposing another learning study; do not retune the completed study.
5. Complete method equations and the code/evidence map in `MATH_ARGUMENT_IMPROVEMENTS_20261103.md` and `METHOD_MATH_AUDIT_20261105.md`. Follow `FRONTIER_GAP_UPDATE_20261105.md` for the 2022–2026 gap.
6. Produce separate-cohort curves, fixed-final bars and A/Cevt/J timelines. The test namespace is reserved and unevaluated; freeze a confirmation/inference plan before any test evaluation. Standard affine constant-fill equivalence is an offline diagnostic, not a new theorem or evidence of equivalent training paths.
7. The separately frozen [offline actor-response diagnosis](NEAR_GOAL_ACTOR_OFFLINE_RESULTS_20261105.md) is complete on the same 6144 replay observations. Near-goal marginal MC-minus-deterministic membership differences have three positive/three negative histories; most states are excluded from one-step capture by a necessary distance bound. These are correlated replay-state kinematic responses, not task success, control failure or final on-policy occupancy. Next quantify dwell and source-derived reward components on the already saved 72 final trajectories, explicitly marking the missing evaluation-return reference UNAVAILABLE. No training/entropy change is implied.

---

## 5. Writing timeline (Asia/Shanghai)

| When | Deliverable | Owner note |
|------|-------------|------------|
| **2026-11-04** | This plan + frontier related-work + outline → GitHub | Done this commit |
| **2026-11-05** | Final suite table, source-matched equations, failure audit and Windows hardware record | Final table/temperature sign/Markov/discount corrections |
| **2026-11-05 complete** | English Methods, B 40k fixed endpoints and final settling-event diagnosis | Original advancement rule met for planning only; both absolute gates FAIL |
| **+1 week** | Related Work v2 (cite frontier table; cut AoI unless needed) | |
| **+2 weeks** | Experiments + ablations draft with figures | |
| **By mid-Nov 2026** | Internal full draft (EN) for WoWMoM shape | |
| **Before 2026-12-01 23:59 AoE** | Anonymized PDF + artifact README freeze | Beijing 2026-12-02 19:59; official CFP recheck |
| **If delayed** | WIP track or journal/robotics fallback | |

---

## 6. Doc inventory (what already exists)

| Path | Role |
|------|------|
| `docs/MATH_ARGUMENT_IMPROVEMENTS_20261103.md` | Equation spine E1–E7 |
| `docs/MASAC_V4_MANEUVER_PAPER_SUITE_PROTOCOL_20261103.md` | Suite contract |
| `docs/MASAC_V4_MANEUVER_PAPER_SUITE_STATUS_20261103.md` | Live gates |
| `docs/MASAC_V4_MANEUVER_ZERO_GATE_20261103.md` | Zero FAIL write-up |
| `docs/TRAINING_OPTIMIZATION_MANEUVER_20261103.md` | Training opts |
| `docs/PENDING_LAUNCHES_MANEUVER_NEXT_ITER_20261103.md` | Next A/B ideas |
| `docs/design/MASAC_V4_*.md` | Architecture / lit / research spec |
| `research/PAPER_MATRIX.md` | Older paper matrix |
| **This file** | Story / C1–C3 / timeline |
| `docs/RELATED_WORK_FRONTIER_20261104.md` | Frontier survey |
| `docs/PAPER_OUTLINE_DRAFT.md` | Section outline |
| `docs/PAPER_METHOD_DRAFT_20261105.md` | Source-matched English Methods with completed 40k status |
| `docs/MATH_CONSTANT_FILL_IDENTITY_20261105.md` | Standard affine identity and optimizer-path limitations |
| `results/masac_v4_sync_40k_analysis_20261105/analysis.md` | Canonical final 40k endpoint and original decision |

---

## 7. Git / process rules

- All paper docs commit + push to GitHub (`research/wowmom2027-upgrade` and `main`).
- The historical Mac pause was superseded within the explicitly authorized Windows audit/B scope. CPU replication, benchmark-supported CUDA pilot and the registered 40k study are complete; historical CPU results remain a separate cohort. Completion and the advancement rule do not automatically launch confirmation or open the sealed maps.
- No Attention, L1 or complete suite rerun is authorized. Large launches require a visible plan summary first. Do not infer authorization for unrelated changes from this migration.
- Preserve saved suite outputs and unrelated work; no force-push. Use the hardware audit to set threads and concurrency; stream commands, evaluations, failures and checkpoint paths.
