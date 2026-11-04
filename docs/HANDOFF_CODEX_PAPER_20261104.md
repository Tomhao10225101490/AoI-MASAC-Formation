# HANDOFF — Codex paper research takeover (2026-11-04)

**Audience:** Desktop Codex (or any coding agent continuing this project).  
**Author pause:** User paused further Grok-Bot experiment launches so Codex can take over paper research.  
**Handoff time:** 2026-11-04 ~22:51 Asia/Shanghai (CST, UTC+8).  
**Machine idle at handoff:** yes — no `train_masac` / maneuver suite / MASAC train PIDs on TomMac-mini.

This file is self-contained. Prefer it over chat history.

---

## 1. Goal

**Predictive Maneuverability-Aware MASAC** for **3-UAV joint capture**.

- Setting: multi-agent SAC (MASAC v4) on formation / joint-capture env.
- Core claim under test: does a **predictive maneuverability probe** (19×2s horizon + RiskEncoder / CV channels) improve 3-agent joint capture vs matched-capacity controls?
- Paper target venue (working): **IEEE WoWMoM 2027**.
- Ablation arms already run: **full** (predictive) vs **static** (static margins / non-predictive probe content) vs **zero** (RiskEncoder capacity, zeroed / no useful probe content).
- Floor beneath the suite: Path-C reward reshape + Standard-C **L0 frozen capture 4/2 latch=False** multi-seed PASS.

Honest outcome stance is allowed: suite already shows **mixed / negative** for “prediction helps” under this protocol — **static ≫ full ≫ zero**. Paper should report that cleanly, not force a positive story.

Repo: https://github.com/Tomhao10225101490/masac-maneuverability  
Local project root on Mac mini: `/Users/a1/Desktop/research/MASAC`

---

## 2. Hard rules (do not violate)

1. **All work local on TomMac-mini**  
   - machineId: `c70e463e-5629-4769-84af-f0ba33183f85`  
   - hostname: `TomMac-mini.local`  
   - **No Cursor CloudAgent** / no off-box training.

2. **After each meaningful stage: commit + push to GitHub**  
   - Branches in use: `main` and `research/wowmom2027-upgrade`  
   - Prefer documenting gates/results/paper drafts in `docs/` and pushing both tips when they should stay aligned.

3. **`do_not_auto_launch_further` until explicit user/Codex-owner authorization**  
   - Suite contract forbids silent next-stage launch.  
   - User **paused** next experiments on 2026-11-04 ~22:51 Asia/Shanghai for Codex takeover.  
   - **Do NOT launch training** until the user (or this handoff’s owner) explicitly re-authorizes a named plan (A/B/C/D or other).

4. **Attention encoder and L1 maps require new explicit auth** — deferred; do not start.

5. **Do not kill unrelated processes** as part of “cleanup” unless the user asks. At handoff the train suite is already finished and the machine is idle.

6. Working tree note: Mac mini checkout may have **many unrelated local dirty/untracked files** (algos/, envs/, audit scripts, continue configs). **Do not bulk-commit them** into the paper handoff. Add only intentional paper/science files.

---

## 3. Repo / branch tips (verify)

At handoff write time (after `git fetch`):

| Ref | Tip | Subject |
|-----|-----|---------|
| `origin/main` | **`c28836a`** | docs: maneuver suite COMPLETE — static FAIL 2/3 closes stamp 162345 |
| `origin/research/wowmom2027-upgrade` | **`c28836a`** | same |
| local `research/wowmom2027-upgrade` | `c28836a` | tracking origin |
| local `main` | may lag until ff-merge; **trust `origin/main`** | — |

Verify before editing:

```bash
cd /Users/a1/Desktop/research/MASAC
git fetch origin --prune
git log -1 --oneline origin/main
git log -1 --oneline origin/research/wowmom2027-upgrade
```

This handoff file itself is a new commit **on top of** `c28836a` (see git log after pull).

---

## 4. Pipeline already done

Chronology (Asia/Shanghai):

1. **Path-C reward reshape** (empty / sanity path) → gate docs under `docs/MASAC_V4_C_RESHAPE_*`.
2. **Empty / C reshape empty** → PASS (see `docs/MASAC_V4_C_RESHAPE_EMPTY_GATE_20261103.md`).
3. **L0 frozen capture 4/2 latch=False** multi-seed → **PASS 3/3**  
   - Finals: **11/12, 11/12, 10/12** (seeds 911 / 922 / 933 × 50k)  
   - Out: `results/masac_v4_c_reshape_l0_frozen42_20261103_145043`  
   - Gate: `docs/MASAC_V4_C_RESHAPE_L0_FROZEN42_GATE_20261103.md`  
   - Finished ~2026-11-03 15:41 Asia/Shanghai.
4. **Predictive Maneuverability paper suite** stamp **`20261103_162345`**  
   - Three arms: full / static / zero  
   - Contract: C reshape + frozen capture **4/2 latch=False**; cold start; gate **≥0.60** each seed on 12 L0 maps; V4 pool + RiskEncoder capacity matched.  
   - Seeds: **1211, 1222, 1233** × **80k** (per arm).  
   - **COMPLETE** 2026-11-04 **18:34** Asia/Shanghai (~26 h wall, three-arm).  
   - Status: `docs/MASAC_V4_MANEUVER_PAPER_SUITE_STATUS_20261103.md`  
   - Protocol: `docs/MASAC_V4_MANEUVER_PAPER_SUITE_PROTOCOL_20261103.md`

---

## 5. Suite results (FINAL)

Gate threshold: ≥0.60 success rate ⇒ **≥8/12** per seed. Suite-level arm gate: count of seeds PASS / 3.

| Arm | Arm gate | Finals (1211, 1222, 1233) | Notes |
|-----|----------|---------------------------|-------|
| **zero** | **FAIL 0/3** | **1/12, 0/12, 1/12** | Capacity w/o probe content collapses |
| **full** | **FAIL 1/3** | **3/12, 4/12, 8/12** | Only 1233 PASS; mid best **9/12 @72k** on 1233 |
| **static** | **FAIL 2/3** | **11/12, 10/12, 5/12** | Strongest; 1233 late collapse **9/12 → 5/12** (best mid **9/12 @70k**) |

**Ablation rank:** **static ≫ full ≫ zero**.

Per-arm gate docs (+ JSON):

- `docs/MASAC_V4_MANEUVER_ZERO_GATE_20261103.md`
- `docs/MASAC_V4_MANEUVER_FULL_GATE_20261103.md`
- `docs/MASAC_V4_MANEUVER_STATIC_GATE_20261103.md`

Qualitative pattern across arms: **ever_ball** often reaches 12/12, but **joint / sync capture** fails — historically `timeout_after_ball` / near-goal uncaptured, not collisions.

---

## 6. Key result directories

Paper suite (stamp `20261103_162345`):

```
results/masac_v4_maneuver_paper_suite_20261103_full_20261103_162345
results/masac_v4_maneuver_paper_suite_20261103_zero_20261103_162345
results/masac_v4_maneuver_paper_suite_20261103_static_20261103_162345
```

L0 floor:

```
results/masac_v4_c_reshape_l0_frozen42_20261103_145043
```

Also look for per-seed gate diagnosis dirs named like:

- `results/masac_v4_c_reshape_l0_frozen42_seed{911,922,933}_gate_20261103/`
- suite per-arm seed eval folders under each suite out dir

---

## 7. Paper / science docs already on GitHub

Primary paper pack (written 2026-11-04, then suite numbers finalized):

| Doc | Role |
|-----|------|
| `docs/PAPER_PLAN_WOWMOM2027.md` | Venue plan, claim, ablation design |
| `docs/RELATED_WORK_FRONTIER_20261104.md` | Frontier related-work table (~18 papers) |
| `docs/PAPER_OUTLINE_DRAFT.md` | Section outline; fill finals after suite |
| `docs/MATH_ARGUMENT_IMPROVEMENTS_20261103.md` | Soft Bellman vs success predicates hygiene |
| `docs/TRAINING_OPTIMIZATION_MANEUVER_20261103.md` | Vectorized probes, `cache_risk`, `--torch-threads 1` |
| `docs/PENDING_LAUNCHES_MANEUVER_NEXT_ITER_20261103.md` | **Next-iter options A/B/C/D — NOT authorized to launch** |
| `docs/PENDING_LAUNCHES_MANEUVER_PAPER_SUITE_20261103.md` | Suite launch pending history |
| `docs/MASAC_V4_MANEUVER_*_GATE_20261103.md` | zero / full / static gates |
| `docs/MASAC_V4_MANEUVER_PAPER_SUITE_STATUS_20261103.md` | Suite COMPLETE status |
| `docs/MASAC_V4_MANEUVER_PAPER_SUITE_PROTOCOL_20261103.md` | Protocol |
| `docs/MASAC_V4_C_RESHAPE_L0_FROZEN42_GATE_20261103.md` | L0 PASS floor |
| **`docs/HANDOFF_CODEX_PAPER_20261104.md`** | **This file** |

Update paper outline / plan with the **final** suite table in §5 (static 2/3, full 1/3, zero 0/3) if any draft still says “waiting on 1233”.

---

## 8. Scientific reading (current)

1. **Observation-only predictive probes are not yet beating static margins** under matched capacity + frozen 4/2 latch=False L0 gate. Full (1/3) loses to static (2/3).
2. **Zero collapses (0/3)** → encoder **capacity without probe content** is insufficient; the static/full content channels matter, but **prediction** is not the winner.
3. **Joint-capture / sync sparsity is the bottleneck** — agents get ball (ever_ball high) but fail simultaneous capture (`timeout_after_ball` class). This motivates curriculum / sync shaping more than “add more prediction channels” alone.
4. Static seed **1233 late collapse** (9/12@70k → 5/12@80k) and full’s sole PASS only on 1233 → variance / instability near gate; paper should show learning curves, not only finals.
5. Negative / mixed result is publishable if framed as a **controlled ablation** of predictive maneuverability features for cooperative capture.

---

## 9. PENDING next-iter options — USER PAUSED

Source of truth: `docs/PENDING_LAUNCHES_MANEUVER_NEXT_ITER_20261103.md`

| Option | Sketch id | Intent | Stance |
|--------|-----------|--------|--------|
| **A** optpack | `masac_v4_maneuver_optpack_20261104` | Throughput only (`cache_risk`, `--torch-threads 1`); no theory claim | Low risk compute appendix |
| **B** sync-capture curriculum | `masac_v4_maneuver_sync_capture_20261104` | Train-time dense sync / occupancy-in-ball shaping; **keep eval** 4/2 latch=False | **Recommended scientific next** |
| **C** risk-aux | `masac_v4_maneuver_risk_aux_20261104` | Aux / IB / shuffle so RiskEncoder cannot be ignored | If arguing probe-use failure |
| **D** +50k continue | new continue id | Warm-start best same-arm ckpt | Needs **explicit** auth |

**USER PAUSED (2026-11-04 ~22:51 Asia/Shanghai).**  
**Await Codex + user re-authorization before any launch.**  
Still forbidden without new auth: Attention, L1, silent budget restack, Cursor cloud.

---

## 10. Suggested Codex next steps (no launch implied)

1. **Tighten paper narrative** with final suite numbers (§5) in `PAPER_PLAN_WOWMOM2027.md` + `PAPER_OUTLINE_DRAFT.md`; keep related-work frontier table current.
2. **If continuing science after user says go:** implement **B** (sync-capture curriculum) as primary method response to static≫full≫zero + joint-capture sparsity; keep L0 eval predicate frozen for apples-to-apples.
3. **Figures / tables from results** — learning curves, finals bar chart, ever_ball vs success, fail-mode breakdown (`timeout_after_ball`); write under `docs/` or `results/.../figures/` and push.
4. **Push everything** to `main` and `research/wowmom2027-upgrade` after each docs/code stage (no force-push).

Optional later (only with auth): A for cleaner wall-time appendix; C if claiming encoder under-use; D only for named continue seeds.

---

## 11. Machine / process checklist for Codex

```bash
# Confirm idle (expect no matches)
pgrep -fl 'train_masac|maneuver_paper|masac_v4' || echo IDLE_NO_PGREP
ps aux | grep -E '[p]ython.*(train|masac|maneuver)' || echo IDLE_NO_TRAIN

# Repo
cd /Users/a1/Desktop/research/MASAC
git status -sb
git log -1 --oneline
```

At handoff: **IDLE_NO_TRAIN / IDLE_NO_PGREP** confirmed on TomMac-mini.

---

## 12. Quick contacts for artifacts

- Suite COMPLETE commit tip before this handoff: **`c28836a`**
- Paper plan pack earlier same day included `dc75940` (plan/related-work/outline) then suite close `3d3be41` / `c28836a` lineage on remotes
- Standing intensity note from prior ops: iterate-from-results was allowed to *prepare* PENDING, but **not** auto-launch further after suite contract

---

**End of handoff.** Codex: read this file, update paper drafts, and **do not start training** until Tom Zhong explicitly re-authorizes a named next experiment.
