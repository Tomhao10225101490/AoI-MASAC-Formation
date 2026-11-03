# Novelty analysis — preliminary, not cleared

Candidate: Maneuverability-Aware MASAC. No novelty claim yet.

| Requested source | Verified entry / current reading status | Necessary comparison |
|---|---|---|
| CycA-MASAC | DOI 10.1007/s10846-026-02418-5 located in search; full mechanism not verified | Verify observation/action model and avoidance mechanism |
| GDC-MASAC | https://www.nature.com/articles/s41598-026-76140-5 publisher entry found | Compare global guidance/decoupled critics; avoid importing formation objective |
| DCRP-ASAC | https://doi.org/10.1109/OJVT.2025.3642719 publisher abstract/method introduction read | Depth/pose/velocity-based dense risk reward is already prior art |
| SCOPE | https://doi.org/10.24963/ijcai.2026/7 proceedings entry found | Need full preview enforcement and safety constraint derivation |

Official source repositories have not yet been confirmed. Search results are not full-text review.
A stopping-distance expression d - v_closing^2/(2 a_relative) is a familiar braking surrogate,
not by itself a new contribution or a general 3D viability certificate. It assumes an achievable
relative acceleration direction and omits coupled obstacle constraints, latency and other agents.
Must compare velocity obstacles, acceleration-aware collision avoidance, CBF feasible-control sets
and reachability/viability methods before defining an equivalent/new mechanism.

Required evidence: matched-capacity observations/auxiliary objective ablations, identical rewards
and budgets where possible, acceleration-limit intervention, independent train seeds and unseen maps.
No inference from DC-RRT speedups or the old sampler training to MASAC effectiveness is valid.

## V3 focused prior-art check

AVO author source https://gamma-web.iacs.umd.edu/AVO/ already addresses acceleration-limited avoidance. SCOPE https://ijcai.org/proceedings/2026/7 already uses preview-fused actor-critic and action correction, with author code https://github.com/98177qdn/SCOPE. DGPPO https://proceedings.iclr.cc/paper_files/paper/2025/hash/b0916c29bc32d57e2c86a63e8f2df9f2-Abstract-Conference.html addresses input constraints. Read V3 design for exact evidence levels.
The narrowed candidate is direction-resolved short-horizon control reserve supplied as a matched-capacity observation. This can still be feature engineering; acceleration sensitivity and direction-shuffling ablations are necessary. The simple V2 braking scalar is superseded as a candidate implementation, not as preserved historical data.
