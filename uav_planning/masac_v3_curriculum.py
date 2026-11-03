"""Training task curriculum for the structured V3 standard MASAC pilot.

All physics, collision checks, rewards and observations are inherited from
StructuredMission.  The first stage changes only the training goals.
"""

import numpy as np

from uav_planning.structured import StructuredMission, box_distance


class CurriculumMission(StructuredMission):
    def __init__(self, *, stage, **kwargs):
        super().__init__(**kwargs)
        if stage not in ("short_edge", "full"):
            raise ValueError(f"Unknown curriculum stage: {stage}")
        self.curriculum_stage = stage

    def reset(self, seed):
        obs = super().reset(seed)
        if self.curriculum_stage == "short_edge":
            self.goal = self.p.copy()
            self.goal[:, 1] += np.where(self.p[:, 1] <= 120.0, 40.0, -40.0)
            if not np.all((self.goal > self.radius) & (self.goal < self.world - self.radius)):
                raise ValueError("Curriculum goal outside world bounds")
            if np.min(box_distance(self.goal, self.lo, self.hi)) <= self.radius:
                raise ValueError("Curriculum goal intersects a building")
            obs = self.observe()
        return obs


def make_episode(seed, episode, completed_steps, cfg):
    """Create a reproducible training episode from a training seed and index.

    completed_steps is counted before reset; a stage never changes during an
    episode. The training map seed and layout cycle follow the original V3 run.
    """
    stages = cfg["stages"]
    if not stages or stages[0]["start_step"] != 0:
        raise ValueError("Curriculum must start at step zero")
    starts = [int(stage["start_step"]) for stage in stages]
    if starts != sorted(set(starts)):
        raise ValueError("Curriculum stage starts must be strictly increasing")
    if episode < 0 or completed_steps < 0:
        raise ValueError("Episode and completed steps must be nonnegative")
    stage = next(stage["name"] for stage in reversed(stages)
                 if completed_steps >= stage["start_step"])
    layout = cfg["layouts"][episode % len(cfg["layouts"])]
    env = CurriculumMission(stage=stage, n=cfg["n_agents"], mode="standard",
                            layout=layout, vmax=cfg["vmax"], amax=cfg["amax"],
                            horizon=cfg["horizon"])
    map_seed = cfg["training_map_seed_offset"] + seed * 1000 + episode
    return env, env.reset(map_seed), stage
