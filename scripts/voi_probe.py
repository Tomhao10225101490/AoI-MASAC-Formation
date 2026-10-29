"""Who should pay altitude for a measurement? Scripted check, no training.

Followers track the slot they believe. A policy decides which followers
leave their slot and climb to buy line of sight over the roofs between them
and the leader. With ``mesh`` on, followers also rebroadcast their leader
belief (envs/formation_env.py, ``follower_relay``), so one climber can serve
the rest of the formation.

Individual policies (each follower decides for itself):

  none       never climb (what the learned residual collapsed to)
  always     always fly ``lift`` metres high (altitude without information)
  std:T      climb ``lift`` metres while the mean slot std exceeds T metres
  aoi:K      climb ``lift`` metres while the age of information exceeds K steps
  los:T      while std exceeds T, climb only to the lowest altitude that sees
             the believed leader over the roofs (map + belief, no oracle)
  voi:L      climb to that altitude only when std * (extra delivery
             probability with LoS) exceeds L * (metres of climb needed)
  oracle     track the true slot (perfect information)
  hold       never climb, last packet instead of the Kalman belief

Prefix any of them with ``mesh+`` to turn forwarding on.

Relay policies (forwarding on, at most one follower climbs):

  relay:fixed     follower 0 (inner row) always holds the leader-LoS altitude
  relay:cheap     the follower with the smallest climb to leader LoS holds it,
                  with hysteresis so the role does not flap
  relay:aoi:K     as cheap, engaged while the freshest belief in the team is
                  older than K steps, held for a dwell after that
  relay:cover...  coverfixed / covercheap / coveraoi:K: the same, but the
                  relay climbs until it sees the leader and every teammate
  relay:team:L    as cheap, but only while the team's information gain
                  (sum of slot std times the relay's LoS delivery
                  probability) exceeds L * (metres of climb)

Formation error is 3D over every follower, the climber included, so the
altitude a relay pays is charged in the same metric the team gains in.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from envs.formation_env import ObsIndex, UAVFormationEnv, env_config_from_dict
from utils.config import load_config

CANYON_EXIT_X = 1040.0
RELAY_HYSTERESIS_M = 15.0
RELAY_DWELL_STEPS = 25
# A follower this far from its slot is more than half a street off: the
# formation is broken for that aircraft.
BREAK_M = 40.0


def _invert_gate(desired: float, nominal: float) -> float:
    """Policy value whose gated mix with ``nominal`` executes ``desired``."""
    if desired >= nominal:
        return 0.0 if nominal >= 1.0 else (desired - nominal) / (1.0 - nominal)
    return 0.0 if nominal <= -1.0 else (desired - nominal) / (1.0 + nominal)


def _los_altitude(env: UAVFormationEnv, i: int, uav, slot_z: float) -> float:
    """Lowest altitude from which the believed leader is in line of sight.

    Uses the map and the belief only, so a follower can compute it on board.
    Returns ``slot_z`` when the believed link is already clear and the
    ceiling when no altitude clears it.
    """
    cfg = env.cfg
    lx, ly, lz = (float(v) for v in env.links[i].belief.mean[:3])
    z = slot_z
    while z < cfg.z_max - 2.0:
        if not env._radio_blocked(lx, ly, lz, uav.x, uav.y, z):
            return z + (5.0 if z > slot_z else 0.0)
        z += 5.0
    return cfg.z_max - 2.0


def _cover_altitude(env: UAVFormationEnv, i: int, uav, slot_z: float) -> float:
    """Lowest altitude that sees the believed leader and every teammate.

    A relay is only worth its climb if it can also reach the followers it
    is supposed to serve. Teammate positions are what the mesh carries.
    """
    cfg = env.cfg
    lx, ly, lz = (float(v) for v in env.links[i].belief.mean[:3])
    mates = [env.uavs[k + 1] for k in range(env.n_followers) if k != i]
    z = slot_z
    while z < cfg.z_max - 2.0:
        if not env._radio_blocked(lx, ly, lz, uav.x, uav.y, z) and not any(
            env._radio_blocked(m.x, m.y, m.z, uav.x, uav.y, z) for m in mates
        ):
            return z + (5.0 if z > slot_z else 0.0)
        z += 5.0
    return cfg.z_max - 2.0


def _los_probability(env: UAVFormationEnv, i: int, uav, z: float) -> float:
    lx, ly, lz = (float(v) for v in env.links[i].belief.mean[:3])
    d = math.dist((uav.x, uav.y, z), (lx, ly, lz))
    return env.ep_p0 * math.exp(-d / env.ep_d0)


def parse_policy(spec: str) -> dict:
    mesh = spec.startswith("mesh+") or spec.startswith("relay:") or spec in ("free",) or spec.startswith("b")
    body = spec[len("mesh+") :] if spec.startswith("mesh+") else spec
    parts = body.split(":")
    kind = parts[0]
    period = 1
    select = "freshest"
    if kind.startswith("b") and kind[1:].isdigit():
        kind = "beacon"
        period = int(parts[0][1:])
        if len(parts) > 1:
            select = parts[1]
    return {
        "spec": spec,
        "mesh": mesh or kind in ("yield", "beacon", "free"),
        "kind": "free" if parts[0] == "free" else kind,
        "mode": parts[1] if parts[0] == "relay" else None,
        "period": period,
        "select": select,
        "threshold": float(parts[-1]) if len(parts) > 1 and parts[-1] not in ("fixed", "cheap", "coverfixed", "covercheap", "random", "round", "freshest") else 0.0,
    }


class ClimbPolicy:
    def __init__(self, env: UAVFormationEnv, policy: dict, lift: float):
        self.env = env
        self.p = policy
        self.lift = lift
        self.relay: int | None = None
        self.dwell = 0

    def reset(self) -> None:
        self.relay = None
        self.dwell = 0

    def _stds(self, observation) -> np.ndarray:
        cfg = self.env.cfg
        return np.array(
            [
                float(np.mean(observation[i + 1][ObsIndex.STD_X : ObsIndex.STD_Z + 1])) * cfg.std_scale
                for i in range(self.env.n_followers)
            ]
        )

    def _targets(self, observation) -> dict[int, float]:
        env, p = self.env, self.p
        kind, thr = p["kind"], p["threshold"]
        if kind in ("none", "oracle", "hold", "yield", "beacon", "free"):
            return {}
        stds = self._stds(observation)
        if kind == "relay":
            return self._relay_target(stds)
        cfg = env.cfg
        out = {}
        for i in range(env.n_followers):
            aoi = float(observation[i + 1][ObsIndex.AOI]) * cfg.aoi_cap
            uav = env.uavs[i + 1]
            sz = env._believed_slot(i)[2]
            if kind == "always":
                out[i] = sz + self.lift
            elif kind == "std" and stds[i] > thr:
                out[i] = sz + self.lift
            elif kind == "aoi" and aoi > thr:
                out[i] = sz + self.lift
            elif kind in ("los", "voi"):
                if kind == "los" and stds[i] <= thr:
                    continue
                z = _los_altitude(env, i, uav, sz)
                if z <= sz + 1.0:
                    continue
                gain = _los_probability(env, i, uav, z) * (1.0 - cfg.nlos_factor)
                if kind == "voi" and stds[i] * gain < thr * (z - sz):
                    continue
                out[i] = z
        return out

    def _relay_target(self, stds: np.ndarray) -> dict[int, float]:
        env, mode = self.env, self.p["mode"]
        n = env.n_followers
        slot_z = [env._believed_slot(i)[2] for i in range(n)]
        if mode in ("fixed", "coverfixed"):
            reach = _cover_altitude if mode == "coverfixed" else _los_altitude
            z = reach(env, 0, env.uavs[1], slot_z[0])
            return {0: z} if z > slot_z[0] + 1.0 else {}
        reach = _cover_altitude if mode.startswith("cover") else _los_altitude
        climbs = np.array(
            [reach(env, i, env.uavs[i + 1], slot_z[i]) - slot_z[i] for i in range(n)]
        )
        if mode in ("aoi", "coveraoi"):
            # Engage when even the freshest belief in the team is stale, then
            # hold the role for a dwell so the relay does not bob up and down
            # every time its own packet resets the age.
            freshest = min(link.aoi for link in env.links)
            if freshest > self.p["threshold"]:
                self.dwell = RELAY_DWELL_STEPS
            elif self.dwell > 0:
                self.dwell -= 1
            if self.dwell <= 0:
                self.relay = None
                return {}
            ranked = climbs.copy()
            if self.relay is not None:
                ranked[self.relay] -= RELAY_HYSTERESIS_M
            self.relay = int(np.argmin(ranked))
            if climbs[self.relay] <= 1.0:
                return {}
            return {self.relay: slot_z[self.relay] + climbs[self.relay]}
        if np.min(climbs) <= 1.0:
            # Someone already sees the leader from its slot: it is the relay
            # for free this step.
            self.relay = int(np.argmin(climbs))
            return {}
        ranked = climbs.copy()
        if self.relay is not None:
            ranked[self.relay] -= RELAY_HYSTERESIS_M
        r = int(np.argmin(ranked))
        if mode == "team":
            z = slot_z[r] + climbs[r]
            gain = _los_probability(env, r, env.uavs[r + 1], z) * (1.0 - env.cfg.nlos_factor)
            if float(np.sum(stds)) * gain < self.p["threshold"] * climbs[r]:
                self.relay = None
                return {}
        self.relay = r
        return {r: slot_z[r] + climbs[r]}

    def __call__(self, observation):
        env = self.env
        actions = np.zeros((env.n_agents, env.action_dim), dtype=np.float32)
        climbing = np.zeros(env.n_followers, dtype=bool)
        for i, target in self._targets(observation).items():
            uav = env.uavs[i + 1]
            sx, sy, sz = env._believed_slot(i)
            ref = env._reference_pose(i)
            target = min(target, env.cfg.z_max - 2.0)
            nominal = float(env._slot_track_action(uav, (sx, sy, sz), ref[4], ref[3])[2])
            desired = float(env._slot_track_action(uav, (sx, sy, target), ref[4], ref[3])[2])
            actions[i + 1, 2] = _invert_gate(desired, nominal)
            climbing[i] = True
        return actions, climbing


def _run(job: dict) -> dict:
    base = load_config(job["config"])
    env_cfg = dict(base["env"])
    env_cfg.update(job["env"])
    policy = parse_policy(job["trigger"])
    env_cfg["follower_mode"] = "slot" if policy["kind"] == "oracle" else "residual"
    obs_mode = "hold" if policy["kind"] == "hold" else "belief"
    env_cfg["obs_mode"] = obs_mode
    env_cfg["leader_jink"] = job["jink"]
    env_cfg["nlos_factor"] = job["nlos_factor"]
    env_cfg["follower_relay"] = False
    env_cfg["canyon_random"] = job["canyon_random"]
    env_cfg["relay_period"] = job["relay_period"]
    env_cfg["beacon_select"] = "freshest"
    if policy["kind"] == "yield":
        # Spend the relay slot only once the freshest packet is older than K.
        env_cfg["relay_mode"] = "tdma"
        env_cfg["relay_aoi_gate"] = int(policy["threshold"])
    elif policy["kind"] == "free":
        env_cfg["relay_mode"] = "free"
    elif policy["kind"] == "beacon":
        env_cfg["relay_mode"] = "beacon"
        env_cfg["relay_period"] = int(policy["period"])
        env_cfg["beacon_select"] = policy["select"]
        env_cfg["relay_aoi_gate"] = -1
    else:
        env_cfg["relay_mode"] = job["relay_mode"] if policy["mesh"] else "off"
        env_cfg["relay_aoi_gate"] = -1
    env = UAVFormationEnv(env_config_from_dict(env_cfg))
    act = ClimbPolicy(env, policy, job["lift"])
    t0 = time.time()
    per_ep = []
    for ep in range(job["episodes"]):
        observation, _ = env.reset(
            seed=job["seed"] + ep,
            options={
                "channel_mode": "distance",
                "p0": job["p0"],
                "randomize_channel": False,
                "use_blockage": job["blockage"],
                "obs_mode": obs_mode,
            },
        )
        act.reset()
        acc = {k: 0.0 for k in ("e3", "eh", "ez", "aoi", "nlos", "climb", "e3_nlos", "n_nlos")}
        step_e3 = []
        n = 0
        steps = 0
        collided = False
        for _ in range(env.cfg.max_steps):
            actions, climbing = act(observation)
            observation, _r, term, trunc, info = env.step(actions)
            steps += 1
            slots = env._true_slots()
            for i, link in enumerate(env.links):
                u = env.uavs[i + 1]
                eh = math.hypot(u.x - slots[i, 0], u.y - slots[i, 1])
                ez = abs(u.z - slots[i, 2])
                e3 = math.hypot(eh, ez)
                acc["e3"] += e3
                step_e3.append(e3)
                acc["eh"] += eh
                acc["ez"] += ez
                acc["aoi"] += link.aoi
                if link.block_class != "los":
                    acc["nlos"] += 1
                    acc["e3_nlos"] += e3
                    acc["n_nlos"] += 1
                acc["climb"] += float(climbing[i])
                n += 1
            collided |= bool(info.get("collision"))
            # Score the canyon only. Once the leader is out and parked on the
            # last waypoint its heading wanders and the slots spin with it.
            if term or trunc or env.uavs[0].x >= CANYON_EXIT_X:
                break
        row = {k: acc[k] / max(n, 1) for k in ("e3", "eh", "ez", "aoi", "nlos", "climb")}
        row["e3_nlos"] = acc["e3_nlos"] / max(acc["n_nlos"], 1)
        e3s = np.asarray(step_e3)
        row["e3_p95"] = float(np.percentile(e3s, 95))
        row["break_frac"] = float(np.mean(e3s > BREAK_M))
        row["relay_per_step"] = env._relay_ok / max(steps, 1)
        row["collision"] = float(collided)
        per_ep.append(row)
    out = {k: job[k] for k in ("jink", "nlos_factor", "blockage", "p0", "trigger")}
    out["relay_mode"] = env.cfg.relay_mode
    out["relay_period"] = int(env.cfg.relay_period)
    out["beacon_select"] = env.cfg.beacon_select
    out["canyon_random"] = bool(env.cfg.canyon_random)
    for key in per_ep[0]:
        vals = np.array([r[key] for r in per_ep])
        out[key] = float(vals.mean())
        out[key + "_sd"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    out["e3_episodes"] = json.dumps([round(r["e3"], 3) for r in per_ep])
    out["seconds"] = time.time() - t0
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/paper_belief_masac.yaml")
    parser.add_argument("--out", default="results/voi")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lift", type=float, default=50.0)
    parser.add_argument("--p0-values", default="1.0,0.6,0.3")
    parser.add_argument("--jinks", default="0.0,0.6")
    parser.add_argument("--nlos-factors", default="0.2")
    parser.add_argument("--relay-mode", default="tdma", choices=("off", "free", "tdma"))
    parser.add_argument("--relay-period", type=int, default=2)
    parser.add_argument("--canyon-random", action="store_true")
    parser.add_argument(
        "--triggers",
        default="oracle,hold,none,always,std:10,std:25,aoi:10,los:0,los:4,los:8,los:15,los:25",
    )
    parser.add_argument("--no-blockage-control", action="store_true")
    args = parser.parse_args(argv)

    env_over = {
        "scenario": "canyon",
        "leader_mode": "route",
        "end_on_collision": False,
        "max_steps": 750,
        "leader_jink_yaw": 0.0,
        "leader_jink_hold": 10,
        "randomize_wind": False,
        "wind_speed": 0.0,
        "canyon_random": args.canyon_random,
    }
    blockages = (True,) if args.no_blockage_control else (True, False)
    jobs = [
        {
            "config": args.config,
            "env": env_over,
            "jink": jink,
            "nlos_factor": nlos,
            "blockage": blockage,
            "p0": p0,
            "trigger": trigger,
            "relay_mode": args.relay_mode,
            "relay_period": args.relay_period,
            "canyon_random": args.canyon_random,
            "lift": args.lift,
            "episodes": args.episodes,
            "seed": args.seed,
        }
        for jink, nlos, blockage, p0, trigger in itertools.product(
            [float(j) for j in args.jinks.split(",")],
            [float(f) for f in args.nlos_factors.split(",")],
            blockages,
            [float(p) for p in args.p0_values.split(",")],
            args.triggers.split(","),
        )
    ]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    start = time.time()
    with Pool(args.workers) as pool:
        for n, row in enumerate(pool.imap_unordered(_run, jobs), 1):
            rows.append(row)
            print(
                f"[{n:3d}/{len(jobs)}] {row['relay_mode']:4s} 机动={row['jink']:.1f} 穿透={row['nlos_factor']:.2f} "
                f"遮挡={'开' if row['blockage'] else '关'} "
                f"p0={row['p0']:.1f} {row['trigger']:14s} "
                f"3D误差={row['e3']:5.1f}±{row['e3_sd']:4.1f} 水平={row['eh']:5.1f} 垂直={row['ez']:5.1f} "
                f"AoI={row['aoi']:4.1f} NLOS占比={row['nlos']:.2f} 爬升占比={row['climb']:.2f} "
                f"转发/步={row['relay_per_step']:.2f} 碰撞={row['collision']:.2f}  ({time.time() - start:.0f}s)",
                flush=True,
            )
    rows.sort(key=lambda r: (r["jink"], -r["nlos_factor"], not r["blockage"], -r["p0"], r["trigger"]))
    with (out / "voi.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
