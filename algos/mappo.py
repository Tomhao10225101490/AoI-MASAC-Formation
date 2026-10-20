"""MAPPO placeholder.

MAPPO is on-policy. It does not share the replay buffer or the update loop
used by MASAC / MADDPG / MATD3, so it is intentionally not faked here.
Do not put MAPPO numbers in the paper until this class is replaced with a
real implementation and the smoke test is extended to cover it.

What a faithful implementation in this repo should look like
-----------------------------------------------------------
1. Keep the same environment observation. Followers still see the stale or
   AoI-aware leader features selected by ``obs_mode``. MAPPO does not remove
   the need for the channel model.
2. One decentralized actor per agent (parameter sharing across followers is
   optional). Action distribution can be a squashed Gaussian, same as SAC,
   but trained with a PPO clipped surrogate rather than the entropy-regularized
   Q objective.
3. One centralized value ``V(global_state)``. This is the CTDE critic. It
   should read ``env.global_state()``, not the concatenated local observations,
   so it matches the privileged state the other algorithms' critics see.
4. Roll out full episodes (or fixed-length chunks), compute GAE(lambda)
   advantages, normalize them, and optimize the clipped objective plus a
   value loss. Keep a separate on-policy buffer. Do not push those
   transitions into ``ReplayBuffer``.
5. Report the same episode metrics as the other algorithms so
   ``scripts/sweep_packet_loss.py`` can consume the checkpoint without a
   special case.

Until then, constructing this class raises ``NotImplementedError``.
"""

from __future__ import annotations


class MAPPO:
    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs
        raise NotImplementedError(
            "MAPPO is a TODO stub, not a trainable baseline yet.\n"
            "The off-policy baselines (MASAC family, MADDPG, MATD3) are "
            "implemented. MAPPO needs its own on-policy loop: decentralized "
            "actors, a centralized V(global_state), GAE advantages, and PPO "
            "clipping. See the docstring in algos/mappo.py for the intended "
            "design, and do not report MAPPO numbers from this repository "
            "until that design is actually trained and smoke-tested."
        )
