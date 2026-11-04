# PAPER_PLAN_WOWMOM2027 — Predictive Maneuverability-Aware MASAC

**Created:** 2026-11-04 15:20 Asia/Shanghai  
**Repo:** https://github.com/Tomhao10225101490/masac-maneuverability  
**Working branch:** `research/wowmom2027-upgrade` (keep `main` in sync)  
**Status:** Planning + frontier survey + outline committed; **do not claim novelty until suite gates close.**

---

## 0. One-sentence story

We study whether **observation-only predictive maneuverability probes** (fixed acceleration directions + short-horizon CV clearance / TTC / braking features, fused by a RiskEncoder) help multi-UAV MASAC learn **joint capture under dynamics**, and we **ablate** predictive content (`full` / `static` / `zero`) against a **Standard-C L0** Path-C floor under a frozen capture contract.

---

## 1. Problem & framing (WoWMoM-facing)

**Task.** \(N=3\) UAVs, continuous 3D acceleration actions, independent goals on frozen L0 city maps; success = **joint** capture (all agents in ball with speed limit, latch off).

**Why it is hard.** Agents can reach balls individually (`ever_ball`↑) yet fail synchronized capture (`timeout_after_ball`). Representation of *feasible near-future motion relative to peers/obstacles* may matter more than raw geometry alone.

**Why WoWMoM / mobile AI systems.** Fits AI/ML for mobile cyber-physical agents, navigation/mobility management, and reproducible open artifacts — **not** a radio/AoI paper unless a later extension adds communication freshness. Keep AoI only as adjacent related work.

**Venue notes (IEEE WoWMoM 2027).**
- Event: Irvine, CA, **21–24 June 2027** ([CFP](https://wowmom2027.dii.unipi.it/call-for-papers.html)).
- WikiCFP lists main deadline **2026-12-01 AoE**, notify ~2027-02-10, camera-ready ~2027-05-05 (verify on official site before submit).
- Double-blind; regular + WIP tracks; IEEE two-column.
- Align narrative to: AI-assisted mobile systems, localization/navigation/mobility, reproducible artifacts — **de-emphasize** pure control-theory novelty if venue reviewers expect networking/systems angle.
- Fallback venues if WoWMoM fit is weak after results: ICRA / IROS / Drones journal / IEEE T-ITS short — decide after suite closes.

---

## 2. Contributions (draft C1–C3 — claim only after evidence)

| ID | Draft claim | Evidence needed before claiming |
|----|-------------|----------------------------------|
| **C1** | **Predictive maneuverability probes + RiskEncoder** as a *representation* for continuous multi-UAV MASAC (not a hard safety filter). | `full` vs parameter-matched `zero` under identical Path-C / capture / budget. |
| **C2** | **Ablation of predictive content:** static geometry channels vs full CV peer/moving channels vs zero — shows *what* in the probe matters. | `static` vs `full` vs `zero` paired seeds 1211/1222/1233. |
| **C3** | **Reproducible Path-C + frozen capture protocol** establishing a Standard L0 learnability floor, then isolating the encoder upgrade. | Standard-C L0 gate already PASS; suite cold-start documented. |

**Non-claims (explicit).**
- Probes are **not** continuous-time certificates, not joint multi-agent safety filters, not differential-flatness planners.
- Path-C reward reshape is **engineering curriculum**, not scientific novelty.
- Do **not** claim parameter-matched vs Legacy Standard-C encoder.
- Do **not** claim Attention / L1 / communication until separately gated.

---

## 3. Ablation map (paper experiments spine)

Shared contract: C reshape; capture **4 m / 2 m·s⁻¹**, `latch=False`; cold start; **80k×3** seeds **1211/1222/1233**; gate ≥0.60 (8/12); L0 maps `830001`–`830012`.

| Arm / baseline | Role | Live result (2026-11-04 Asia/Shanghai) |
|----------------|------|----------------------------------------|
| **Standard-C L0** (`c_reshape_l0_frozen42`) | Pre-innovation floor (legacy encoder) | **PASS** 11/12, 11/12, 10/12 (seeds 911/922/933, 50k) |
| **maneuver-zero** | Capacity-matched control (zeros → RiskEncoder) | **COMPLETE FAIL** 1/12, 0/12, 1/12; best ~5/12 |
| **maneuver-static** | Static geometry probes only (mv+peer channels zeroed) | **1211 PASS 11/12**, **1222 PASS 10/12**; **1233 TBD** (mid ~8/12 @44k) |
| **maneuver-full** | Full CV probes (candidate) | **1211 FAIL 3/12**, **1222 FAIL 4/12**; **1233 TBD** (weak ~1–2/12 mid) |
| shuffle (optional) | Direction–feature permutation control | Not launched |

**Working narrative implication (provisional, not final claim):**
- RiskEncoder capacity alone is insufficient (`zero` collapses).
- Static predictive geometry appears learnable under this contract (`static` strong so far).
- Full dynamic/peer channels are **not yet helping** and may hurt under current feature scale / curriculum — treat as **negative or mixed result** until seed 1233 + next-iter opts finish. Paper can still be strong if we honestly report when predictive dynamics fail and what static margins buy.

---

## 4. Results we still need before drafting “claims”

1. Close **full** and **static** seed **1233** gates (do not kill training).
2. Tabulate per-seed: joint success, ever_ball, timeout_after_ball, collision, path efficiency.
3. Decide primary story branch:
   - **A (preferred if static PASS & full weak):** “Static maneuverability margins help; naive full CV peer channels need further design.”
   - **B (if full recovers):** “Full predictive probes outperform static/zero.”
   - **C (if both fail vs Standard-C):** “Architecture cold-start + capacity change; revisit warm-start / optpack before claiming C1.”
4. Optional next-iter (PENDING only; no auto-launch): sync-capture curriculum, feature scaling, optpack — see `docs/PENDING_LAUNCHES_MANEUVER_NEXT_ITER_20261103.md`.
5. Figures: learning curves (3 seeds), ablation bar chart, failure-mode pie (`timeout_after_ball` vs collision), probe feature schematic.

---

## 5. Writing timeline (Asia/Shanghai)

| When | Deliverable | Owner note |
|------|-------------|------------|
| **2026-11-04** | This plan + frontier related-work + outline → GitHub | Done this commit |
| **Suite close (±1–2 days)** | Fill numbers in outline; freeze ablation table | After full/static complete |
| **+3–5 days** | Method section draft from `MATH_ARGUMENT` E1–E7 | Equations already audited |
| **+1 week** | Related Work v2 (cite frontier table; cut AoI unless needed) | |
| **+2 weeks** | Experiments + ablations draft with figures | |
| **By mid-Nov 2026** | Internal full draft (EN) for WoWMoM shape | |
| **Before 2026-12-01 AoE** | Anonymized PDF + artifact README freeze | Verify deadline on CFP |
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
- No Cursor cloud agents; train only on TomMac-mini; **do not stop** suite PIDs.
- `do_not_auto_launch_further=true` until user authorizes next stage.
