#!/usr/bin/env python3
"""Authorized V4 L0 continuation (+extra-steps) without mutating the frozen runner.

Keeps scripts/train_masac_v4_l0.py byte-identical so parent checkpoint source
hashes still verify via restore_snapshot. Training loop body matches the frozen
runner; only experiment labeling, seed selection (resumed seed only), and the
target step budget (checkpoint_step + extra_steps) differ.

Authorization: user B 2026-11-03 widget t0s9. failure_action=save_and_diagnose.
No Attention / maneuverability / Critic A/B / L1.
"""
from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_frozen_l0():
    path = ROOT / "scripts/train_masac_v4_l0.py"
    spec = importlib.util.spec_from_file_location("masac_v4_l0_frozen", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def run(args, L0):
    auth = json.loads(args.authorization_protocol.read_text())
    cfg = json.loads(args.protocol.read_text())
    seed_path = ROOT / cfg["seed_sets"]
    sets = json.loads(seed_path.read_text())
    L0.validate_protocol(cfg, sets)
    if args.extra_steps <= 0:
        raise ValueError("--extra-steps must be positive")
    if args.resume is None:
        raise ValueError("--resume required")

    confhash = {"protocol": L0.digest(args.protocol), "seed_sets": L0.digest(seed_path)}
    seeds = cfg["seeds"][:1] if args.smoke else list(cfg["seeds"])
    warmup = 128 if args.smoke else cfg["warmup"]
    batch = cfg["batch_size"]
    capacity = 3000 if args.smoke else cfg["replay_capacity"]
    eval_every = 256 if args.smoke else cfg["eval_interval"]
    save_every = 256 if args.smoke else cfg["checkpoint_interval"]
    eval_maps = cfg["development_maps"][:2] if args.smoke else cfg["development_maps"]
    torch.set_num_threads(1)

    lockpath = ROOT / "results/masac_v4_l0_training.lock"
    lockpath.parent.mkdir(parents=True, exist_ok=True)
    lock = lockpath.open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("Another V4 L0 runner holds the global training lock") from exc

    resume = args.resume.resolve()
    old_root = resume.parent.parent
    old_manifest = json.loads((old_root / "run_manifest.json").read_text())
    if old_manifest["config_hashes"] != confhash or old_manifest["smoke"] != args.smoke:
        raise ValueError("Resume protocol/seed namespace/smoke does not match frozen parent protocol")
    if old_manifest["sources"] != L0.source_hashes():
        raise ValueError("Resume source hash mismatch (frozen train script must be unchanged)")

    args.out.mkdir(parents=True, exist_ok=False)
    root = args.out.resolve()
    snapshot_meta = json.loads((resume / "manifest.json").read_text())
    resume_seed = snapshot_meta["seed"]
    if resume_seed not in seeds:
        raise ValueError("resume seed outside frozen seed set")
    seeds = [resume_seed]
    target_steps = int(snapshot_meta["step"]) + int(args.extra_steps)
    experiment_id = auth.get("experiment_id", "l0_continue_533_plus50k_20261103")

    manifest = {
        "experiment_id": experiment_id,
        "mode": "legacy standard continuation",
        "smoke": args.smoke,
        "level": "L0",
        "created_at": time.time(),
        "pid": os.getpid(),
        "config_hashes": confhash,
        "sources": L0.source_hashes(),
        "seed_list": seeds,
        "parent_steps_per_seed": cfg["steps"],
        "extra_steps": int(args.extra_steps),
        "target_total_steps": target_steps,
        "warmup": warmup,
        "replay_capacity": capacity,
        "development_maps_used": eval_maps,
        "final_test": "reserved and not generated",
        "resume_from": str(resume),
        "parent_run": str(old_root),
        "authorization_protocol": str(args.authorization_protocol.resolve()),
        "authorization": auth.get("authorization"),
        "failure_action": auth.get("failure_action", "save_and_diagnose"),
        "prior_diagnosis": auth.get("parent_run", {}).get("diagnosis_dir"),
        "command": sys.argv,
        "note": "Frozen configs/masac_v4/l0_protocol.json used for hash continuity; +extra-steps is the only budget change.",
    }
    L0.write(root / "run_manifest.json", manifest)
    shutil.copy2(args.authorization_protocol, root / "authorization_protocol.json")
    md = args.authorization_protocol.with_suffix(".md")
    if md.exists():
        shutil.copy2(md, root / "authorization_protocol.md")

    for name in L0.SOURCES:
        target = root / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
        if L0.digest(target) != manifest["sources"][name]:
            raise ValueError("Source changed while taking snapshot")
    for source, rel in (
        (args.protocol, "configs/masac_v4/l0_protocol.json"),
        (seed_path, "configs/masac_v4/seed_sets.json"),
        (args.authorization_protocol, f"configs/masac_v4/{args.authorization_protocol.name}"),
        (Path(__file__).resolve(), "scripts/train_masac_v4_l0_continue.py"),
    ):
        target = root / "source_snapshot" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    if not args.smoke:
        L0.write(
            ROOT / "results/masac_v4_active_run.json",
            {
                "out": str(root),
                "pid": os.getpid(),
                "started_at": time.time(),
                "kind": "v4_l0_continue_533_plus50k",
                "phase": "running",
                "experiment_id": experiment_id,
                "seed": resume_seed,
                "resume_from": str(resume),
                "extra_steps": int(args.extra_steps),
                "target_total_steps": target_steps,
                "authorization": "user B 2026-11-03 widget t0s9",
                "parent_run": str(old_root),
                "log": str(root / "train.log"),
            },
        )

    seed = None
    step = 0
    phase = "initializing"
    completed = []
    begin = time.perf_counter()
    try:
        L0.message(
            root,
            f"L0 CONTINUE | PID={os.getpid()} | seed={resume_seed} | from_step={snapshot_meta['step']} | extra={args.extra_steps} | target={target_steps} | out={root}",
        )
        for seed in seeds:
            directory = root / f"standard_{seed}"
            directory.mkdir()
            model, env, obs, explore, state = L0.restore_snapshot(resume, capacity, cfg["device"])
            if state["seed"] != seed:
                raise ValueError("invalid resume state seed")
            steps = int(state["step"]) + int(args.extra_steps)
            if state["step"] >= steps:
                raise ValueError("resume state already at/beyond target steps")
            L0.message(root, f"START seed={seed} from_step={state['step']} target={steps} | family={env.config.family}")
            start_step = state["step"]
            tick = time.perf_counter()
            for step in range(start_step + 1, steps + 1):
                phase = "training"
                state["step"] = step
                action = explore.uniform(-1, 1, (3, 3)).astype(np.float32) if step <= warmup else model.act(obs)
                next_obs, reward, done, truncated, info = env.step(action)
                model.push(obs, action, reward, next_obs, done)
                obs = next_obs
                state["episode_return"] += reward
                distance = np.linalg.norm(env.goal - env.p, axis=-1)
                speed = np.linalg.norm(env.v, axis=-1)
                state["closest"] = np.minimum(state["closest"], distance).tolist()
                state["near_stall"] = (
                    np.asarray(state["near_stall"]) + ((distance < 10) & (speed < 2) & ~env.captured)
                ).tolist()
                if step > warmup:
                    state["losses"] = model.update(batch)
                    state["updates"] += 1
                    if not all(np.isfinite(v) for v in state["losses"].values()):
                        raise ValueError("nonfinite learning diagnostic")
                if done or truncated:
                    row = {
                        "kind": "episode",
                        "experiment_id": experiment_id,
                        "level": "L0",
                        "seed": seed,
                        "step": step,
                        "episode": state["episode"],
                        "episode_start_step": state["episode_start"],
                        "map_seed": env.seed,
                        "family": env.config.family,
                        "return_": state["episode_return"],
                        "terminated": bool(done),
                        "truncated": bool(truncated),
                        "timeout": bool(truncated and not done),
                        "closest_goal_m_by_uav": state["closest"],
                        "near_goal_low_speed_uncaptured_steps_by_uav": state["near_stall"],
                        **info,
                    }
                    L0.append(directory / "metrics.jsonl", row)
                    state["recent"] = (state["recent"] + [row])[-30:]
                    state["episode"] += 1
                    state["episode_start"] = step
                    state["episode_return"] = 0.0
                    env, obs = L0.audited_episode(
                        cfg["families"][state["episode"] % 6],
                        cfg,
                        cfg["training_map_seed_offset"] + seed * cfg["training_seed_multiplier"] + state["episode"],
                        state["episode"],
                        directory,
                    )
                    state["closest"] = np.linalg.norm(env.goal - env.p, axis=-1).tolist()
                    state["near_stall"] = [0] * 3
                if step % cfg["log_interval"] == 0 or step == steps:
                    recent = state["recent"]
                    elapsed = time.perf_counter() - tick
                    rate = (step - start_step) / elapsed if elapsed else 0.0
                    status = {
                        "phase": phase,
                        "experiment_id": experiment_id,
                        "level": "L0",
                        "seed": seed,
                        "step": step,
                        "total_steps": steps,
                        "parent_checkpoint_step": start_step,
                        "extra_steps": int(args.extra_steps),
                        "updates": state["updates"],
                        "episodes": state["episode"],
                        "family": env.config.family,
                        "steps_per_s": rate,
                        "eta_s": (steps - step) / rate if rate else None,
                        "last_eval": state["last_eval"],
                        "last_eval_step": state["last_eval_step"],
                        "pid": os.getpid(),
                        "updated_at": time.time(),
                        "recent_count": len(recent),
                        "recent_success": sum(x["success"] for x in recent) / len(recent) if recent else None,
                        "recent_terminations": {
                            k: sum(bool(x.get(k)) for x in recent)
                            for k in (
                                "success",
                                "static_collision",
                                "dynamic_collision",
                                "uav_collision",
                                "boundary_collision",
                                "timeout",
                            )
                        },
                        **state["losses"],
                    }
                    L0.write(root / "status.json", status)
                    L0.append(directory / "metrics.jsonl", {"kind": "progress", **status})
                    L0.message(
                        root,
                        f"[L0 continue seed={seed}] {step}/{steps} upd={state['updates']} ep={state['episode']} {rate:.1f}step/s ETA={status['eta_s']}s val={state['last_eval']} q={state['losses'].get('q_mean')} td={state['losses'].get('td_abs')} entropy={state['losses'].get('entropy')}",
                    )
                if step % eval_every == 0 or step == steps:
                    phase = "evaluating"
                    L0.write(
                        root / "status.json",
                        {
                            "phase": phase,
                            "seed": seed,
                            "step": step,
                            "pid": os.getpid(),
                            "experiment_id": experiment_id,
                            "level": "L0",
                            "updated_at": time.time(),
                        },
                    )
                    records = L0.evaluate(model, eval_maps, directory, step, cfg)
                    state["last_eval"] = sum(x["success"] for x in records) / len(records)
                    state["last_eval_step"] = step
                    L0.message(
                        root,
                        f"EVAL seed={seed} step={step}: success={sum(x['success'] for x in records)}/{len(records)}",
                    )
                if step % save_every == 0 or step == steps:
                    phase = "saving"
                    L0.save_snapshot(directory, model, env, explore, state)
            completed.append(
                {
                    "seed": seed,
                    "step": steps,
                    "final_development_success": state["last_eval"],
                    "gate_met": bool(state["last_eval"] >= cfg["gate"]["per_seed_joint_success_fraction"]),
                    "parent_checkpoint_step": start_step,
                    "extra_steps": int(args.extra_steps),
                }
            )
            L0.write(directory / "completion.json", completed[-1])
        complete = {
            "complete": True,
            "scope": "authorized continuation +extra-steps",
            "runs": completed,
            "seeds_expected": seeds,
            "gate_threshold": cfg["gate"]["per_seed_joint_success_fraction"],
            "continuation_gate_met": bool(completed and completed[0]["gate_met"]),
            "failure_action": "save_and_diagnose",
            "smoke": args.smoke,
            "finished_at": time.time(),
            "elapsed_s": time.perf_counter() - begin,
            "authorization": auth.get("authorization"),
        }
        L0.write(root / "completion.json", complete)
        L0.write(root / "status.json", {"phase": "complete", "complete": True, **complete})
        if not args.smoke:
            active = json.loads((ROOT / "results/masac_v4_active_run.json").read_text())
            active.update(
                {
                    "phase": "complete",
                    "completed_at": time.time(),
                    "continuation_gate_met": complete["continuation_gate_met"],
                }
            )
            L0.write(ROOT / "results/masac_v4_active_run.json", active)
        L0.message(root, f"COMPLETE scope={complete['scope']} gate_met={complete['continuation_gate_met']}")
    except BaseException as e:
        L0.write(
            root / "error.json",
            {
                "error": repr(e),
                "traceback": traceback.format_exc(),
                "seed": seed,
                "step": step,
                "phase": phase,
                "updated_at": time.time(),
            },
        )
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    L0 = load_frozen_l0()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/masac_v4/l0_protocol.json")
    parser.add_argument(
        "--authorization-protocol",
        type=Path,
        default=ROOT / "configs/masac_v4/l0_continue_533_plus50k_20261103.json",
    )
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--extra-steps", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args(), L0)
