"""Render a source-aligned paper schematic; no model, environment or data load."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as DrawingPath

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    "uav_planning/v4_networks.py", "uav_planning/v4_prediction.py",
    "uav_planning/v4_maneuver_masac.py", "uav_planning/v4_sync_capture.py",
    "uav_planning/v4_near_objective.py", "uav_planning/v4_near_study_runtime.py",
    "scripts/render_masac_system_figure.py",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render(out):
    source_hashes = {name: digest(ROOT / name) for name in SOURCES}
    out = out.resolve()
    if not out.is_relative_to(ROOT / "docs" / "figures"):
        raise ValueError("Figure output must stay in repository docs/figures")
    out.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.2,
                         "svg.fonttype": "none", "svg.hashsalt": "masac-system-20261105"})
    fig, ax = plt.subplots(figsize=(7.2, 5.0), facecolor="white")
    fig.subplots_adjust(left=.015, right=.985, bottom=.015, top=.985)
    ax.set(xlim=(0, 1.02), ylim=(0, 1.04))
    ax.axis("off")
    ink, blue, teal, gold = "#263445", "#275f94", "#177b70", "#996819"

    def box(x, y, w, h, text, *, edge=ink, fill="#f6f8fa", size=8.2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.008",
                                   linewidth=.85, edgecolor=edge, facecolor=fill, zorder=2))
        ax.text(x+w/2, y+h/2, text, ha="center", va="center", color=ink,
                fontsize=size, linespacing=1.35, zorder=3)

    def arrow(points, *, training=False):
        path = DrawingPath(points, [DrawingPath.MOVETO]+[DrawingPath.LINETO]*(len(points)-1))
        ax.add_patch(FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=8,
                                    linewidth=.95, color=blue if training else ink,
                                    linestyle="--" if training else "-", zorder=1))

    ax.text(.03, 1.015, "Local execution: parameter-shared actor, one action per UAV",
            fontsize=10, weight="bold", color=ink, va="center")
    box(.03, .72, .14, .13, "Current local\nobservation $o_i$\n162 dimensions")
    box(.21, .83, .35, .13, "Ego + masked object-set encoder\nTwo token-wise pool blocks\n384 features", edge=blue, fill="#f0f5fa")
    box(.21, .65, .17, .13, "Fixed probes\n19 $\\times$ 11\nfrom $o_i$ only", edge=teal, fill="#edf7f4")
    box(.41, .65, .15, .13, "Probe MLP\n11 $\\to$ 64 $\\to$ 64\nmean/max: 128", edge=teal, fill="#edf7f4", size=7.9)
    box(.60, .74, .10, .13, "Concat\n512")
    box(.74, .74, .12, .13, "Gaussian\nactor\nsample / tanh", edge=blue, fill="#f0f5fa")
    box(.90, .74, .10, .13, "Discrete\nmotion\n+ limiters")
    arrow([(.17, .785), (.19, .785), (.19, .895), (.21, .895)])
    arrow([(.19, .785), (.19, .715), (.21, .715)])
    arrow([(.38, .715), (.41, .715)])
    arrow([(.56, .895), (.585, .895), (.585, .825), (.60, .825)])
    arrow([(.56, .715), (.585, .715), (.585, .785), (.60, .785)])
    arrow([(.70, .805), (.74, .805)])
    arrow([(.86, .805), (.90, .805)])
    ax.text(.80, .905, "$a_i\\in(-1,1)^3$", ha="center", color=blue, fontsize=8)
    ax.text(.385, .795, "full / static / zero: content masks, equal network width",
            ha="center", color=teal, fontsize=6.8)
    arrow([(.95, .74), (.95, .60), (.012, .60), (.012, .44), (.03, .44)], training=True)
    ax.text(.51, .615, "training collection: $O,A,O'$; cached probes are observation-derived",
            ha="center", color=blue, fontsize=7)
    ax.axhline(.555, xmin=.03, xmax=.98, color="#c0c9d2", linewidth=.7)
    ax.text(.03, .525, "Centralized training: separate actor and critic encoder parameters",
            fontsize=9.5, weight="bold", color=blue, va="center")
    box(.03, .29, .22, .19, "Transition replay\n$(O,A,r_{\\rm train},O',\\delta)$\ncache $\\mathcal{R}(O),\\mathcal{R}(O')$", edge=blue, fill="#f0f5fa")
    box(.30, .32, .29, .16, "Own encoders for each $Q_m$\n$[F_{\\psi_m}(o_i,\\mathcal{R}_i),a_i]$: 515\nPer-agent MLP: 515 $\\to$ 256 $\\to$ 256", size=7.9)
    box(.63, .32, .13, .16, "Agent\nmean + max\n512")
    box(.80, .32, .20, .16, "Separate $Q_1,Q_2$\n512 $\\to$ 256 $\\to$ 1 each\n+ target copies", edge=blue, fill="#f0f5fa")
    arrow([(.25, .39), (.30, .40)], training=True)
    arrow([(.59, .40), (.63, .40)], training=True)
    arrow([(.76, .40), (.80, .40)], training=True)
    ax.text(.65, .275, "SAC updates actor, critics and temperature; $\\delta$ marks true terminals",
            ha="center", color=blue, fontsize=7)
    box(.30, .075, .46, .16, "Training-only reward from the physical transition\nB study: fixed potential addition to Path-C\nNear study: replace near occupancy; B disabled", edge=gold, fill="#faf5e9", size=8)
    arrow([(.30, .155), (.14, .155), (.14, .29)], training=True)
    ax.text(.14, .225, "$r_{\\rm train}$", ha="center", va="center", color=gold,
            fontsize=8, bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    ax.text(.50, .025, "Separate registered studies; evaluation always uses original Path-C. Dashed arrows: training only.",
            ha="center", va="center", color=ink, fontsize=7)

    fig.canvas.draw()
    svg, png = out / "figure1_system.svg", out / "figure1_system.png"
    fig.savefig(svg, bbox_inches="tight", pad_inches=.04,
                metadata={"Date": None, "Creator": "MASAC source-aligned figure renderer"})
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=.04,
                metadata={"Software": "MASAC source-aligned figure renderer"})
    plt.close(fig)
    if any(digest(ROOT / name) != expected for name, expected in source_hashes.items()):
        raise RuntimeError("Illustrated source changed while rendering")
    manifest = dict(complete=True, kind="architecture illustration, not experimental evidence",
                    created_at_utc=datetime.now(timezone.utc).isoformat(),
                    sources=source_hashes, matplotlib_version=matplotlib.__version__,
                    no_model=True, no_environment=True, no_training=True, no_sealed_access=True,
                    manual_source_facts=dict(observation=162, set_features=384, probes=[19, 11],
                        risk_features=128, actor_features=512, action=3, critic_per_agent_input=515,
                        critic_agent_latent=256, critic_team_mean_max=512,
                        critics_have_independent_encoders=True, replay_caches_raw_probes_not_learned_latents=True),
                    artifacts={p.name: dict(sha256=digest(p), bytes=p.stat().st_size) for p in (svg, png)})
    with (out / "illustration_manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    print("ILLUSTRATION COMPLETE no_model=true no_environment=true no_training=true no_sealed_access=true", flush=True)
    for path in (svg, png, out / "illustration_manifest.json"):
        print(f"ARTIFACT {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    render(parser.parse_args().out)
