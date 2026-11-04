# PAPER_PLAN_WOWMOM2027 — Predictive Maneuverability-Aware MASAC

**2026-11-05 evidence update:** All12 archived L0 development maps have zero
external moving objects. Full/static cannot identify moving-obstacle prediction;
new early-replay evidence also exposes neutral-value saturation confounds
(`L0_CONTENT_IDENTIFIABILITY_AUDIT_20261105.md`). The two-seed2k B/control pilot
is complete with0/12 final success in all four runs and finite diagnostics.
The separate fixed40k×3seed static B-off/on protocol tests known potential shaping,
not predictive-content novelty or a full curriculum. See
`design/MASAC_V4_SYNC_CAPTURE_40K_PROTOCOL_20261105.md`.

**Created:** 2026-11-04 15:20 Asia/Shanghai  
**Repo:** https://github.com/Tomhao10225101490/masac-maneuverability  
**Working branch:** `research/wowmom2027-upgrade` (keep `main` in sync)  
**Updated:** 2026-11-05 Asia/Shanghai. Suite COMPLETE; all three arms fail the all-seed gate. Windows audit and B small validation are authorized; large launches require a plan summary first. Prediction superiority, safety and novelty remain unestablished.

---

## 0. One-sentence story

We study an observation-only, fixed-control preview representation for multi-UAV MASAC through a matched-capacity `full` / `static` / `zero` ablation under frozen joint capture. Static-geometry preview has the strongest mean final result; adding CV moving/peer channels has not established a benefit. Standard-C L0 is a separate learnability floor, not a fair performance comparator.

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
| **C3** | A reproducible frozen-capture protocol and a failure audit separating any-UAV entry, all-UAV eventual entry and simultaneous capture. | Saved completion/evaluation records; 61 partial-entry and four all-entry failures. Standard-C supplies only a separate task learnability floor. |

**Non-claims (explicit).**
- Probes are **not** continuous-time certificates, not joint multi-agent safety filters, not differential-flatness planners.
- Path-C reward reshape is **engineering curriculum**, not scientific novelty.
- Do **not** claim parameter-matched vs Legacy Standard-C encoder.
- Do **not** claim Attention / L1 / communication until separately gated.
- Static still previews ego motion for two seconds; full-vs-static is not prediction-vs-no-prediction.
- Zero removes direction coordinates as well as risk scalars; its trainable encoder can learn a constant. Its failure does not prove that capacity is useless or that a learned actor causally uses the risk embedding.
- B sync-capture is a new hypothesis; mathematical shaping identities do not establish learning speed or capture improvement.

---

## 3. Ablation map (paper experiments spine)

Within-suite contract (**full/static/zero only**): C reshape; capture **4 m / 2 m·s⁻¹**, `latch=False`; cold start; **80k×3** seeds **1211/1222/1233**; gate ≥0.60 (8/12); L0 maps `830001`–`830012`.

| Arm / baseline | Role | Final result (suite completed 2026-11-04) |
|----------------|------|----------------------------------------|
| **Standard-C L0** (`c_reshape_l0_frozen42`) | Separate learnability floor; legacy encoder and warm start | **PASS 3/3**: 11/12, 11/12, 10/12 (seeds 911/922/933, 50k L0) |
| **maneuver-zero** | Matched-parameter constant-risk-input control | **FAIL 0/3**: 1/12, 0/12, 1/12; best ~5/12 |
| **maneuver-static** | Static/boundary geometry preview; excludes mv+peer from derived aggregates | **FAIL 2/3**: 11/12, 10/12, 5/12; seed 1233 falls from 9/12 at 70k to 5/12 |
| **maneuver-full** | Full CV probes (candidate) | **FAIL 1/3**: 3/12, 4/12, 8/12; seed 1233 alone passes |
| shuffle (optional) | Direction–feature permutation control | Not launched |

**Evidence boundary:** Standard-C loads the same empty-floor seed-811 40k checkpoint for each L0 seed, retaining weights and optimizers while resetting replay/RNG. Different architecture, initialization history, seeds and 50k L0 budget confound comparison with the 80k cold-start maneuver arms.

Descriptive final totals are static **26/36**, full **15/36**, zero **2/36**. These are three training seeds evaluated on the same 12 development maps, not 36 independent training trials. No arm passes all three seeds. Preserve finals and learning curves; do not replace seed 1233's final static value by its best checkpoint. No sealed unseen-map generalization claim is available.

---

## 4. Results we still need before drafting “claims”

1. Freeze the completed three-seed finals above; carry seed variability and the static 1233 drop into the abstract and results.
2. Report joint success, **any** entry, **all eventual** entry, collision and success-conditional path efficiency separately. All-eventual entry counts: static **11,10,5**; full **5,4,8**; zero **1,1,2**.
3. Use the mixed/negative story: static-geometry preview is strongest on average, but the value of CV moving/peer channels remains unestablished. Correlations and final ranks do not identify a failure cause.
4. Implement B as the smallest response to worst-agent approach/capture failures. Follow `docs/design/MASAC_V4_SYNC_CAPTURE_20261105.md` with a new id and unchanged evaluation; first run a small control/B validation after the Windows audit. Summarize the plan before any large training.
5. Complete method equations and the code/evidence map in `MATH_ARGUMENT_IMPROVEMENTS_20261103.md` and `METHOD_MATH_AUDIT_20261105.md`. Follow `FRONTIER_GAP_UPDATE_20261105.md` for the 2022–2026 gap.
6. Produce three-seed curves, final bars, per-UAV approach/capture timelines and partial-entry/all-entry failure breakdowns. Reserve a sealed test set before confirmation.

---

## 5. Writing timeline (Asia/Shanghai)

| When | Deliverable | Owner note |
|------|-------------|------------|
| **2026-11-04** | This plan + frontier related-work + outline → GitHub | Done this commit |
| **2026-11-05** | Final suite table, source-matched equations, failure audit and Windows hardware record | Final table/temperature sign/Markov/discount corrections |
| **+3–5 days** | Method draft and B small-validation analysis | Keep implementation, algebra and empirical evidence separate |
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

---

## 7. Git / process rules

- All paper docs commit + push to GitHub (`research/wowmom2027-upgrade` and `main`).
- The 2026-11-04 Mac pause is historical. The 2026-11-05 user request authorizes local Windows auditing, CPU-protocol replication and B implementation/small validation. Keep CPU as the default until a CUDA benchmark establishes a stable speed benefit.
- No Attention, L1 or complete suite rerun is authorized. Large launches require a visible plan summary first. Do not infer authorization for unrelated changes from this migration.
- Preserve saved suite outputs and unrelated work; no force-push. Use the hardware audit to set threads and concurrency; stream commands, evaluations, failures and checkpoint paths.
