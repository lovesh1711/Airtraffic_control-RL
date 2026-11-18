"""Permutation-invariant set encoder (features extractor for SB3).

Intruders, obstacles and probe candidates are each a set: every entity goes
through a small shared MLP and the results are pooled by attention, with a
query computed from the rest of the observation (the context). Slot order
then does not matter, a variable number of entities is handled with the
validity masks, and the same weights serve 10 intruders or 9 neighbours.

The extractor works on the flattened observation; the index lists that pick
each set out of it are computed from the Dict space.
"""

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from core.observations import WaypointObservation, OwnAirspeedObservation
from solution.observations import (RiskIntruderObservation, ThreatObstacleObservation,
                                   ThreatBoundaryObservation, RouteObservation)
from solution.probe import ManoeuvreProbeObservation

INTRUDER_KEYS = ("x_r", "y_r", "vx_r", "vy_r", "cos_track", "sin_track",
                 "intruder_distance", "tcpa", "dcpa")
INTENT_KEYS = ("tcpa_plan", "dcpa_plan")
SHARE_KEYS = RiskIntruderObservation.SHARE_KEYS
MASK_KEY = "intruder_valid"

OBSTACLE_KEYS = ("obstacle_x", "obstacle_y", "obstacle_radius", "obstacle_clearance",
                 "obstacle_t_enter", "obstacle_miss", "obstacle_inside")
OBSTACLE_MASK_KEY = "obstacle_valid"

PROBE_KEYS = ManoeuvreProbeObservation.FEATURES
PROBE_MASK_KEY = ManoeuvreProbeObservation.MASK_KEY


def intruder_keys(intent=False, share_intent=False):
    return INTRUDER_KEYS + (INTENT_KEYS if intent else ()) + (SHARE_KEYS if share_intent else ())


def competition_obs_space(n_intruders=10, n_obstacles=5, n_sector=12, intent=False,
                          route_obs=False, probe=False, share_intent=False):
    """The Dict observation space, built without a simulator."""
    return spaces.Dict({
        **WaypointObservation(n=1, distance_norm=170.0).space(),
        **OwnAirspeedObservation(spd_mean=150.0, spd_std=20.0).space(),
        **RiskIntruderObservation(n=n_intruders, intent=intent, share=share_intent).space(),
        **ThreatObstacleObservation(n=n_obstacles).space(),
        **ThreatBoundaryObservation(n=n_sector, distance_norm=170.0).space(),
        **(RouteObservation().space() if route_obs else {}),
        **(ManoeuvreProbeObservation().space() if probe else {}),
    })


def flat_slices(space):
    """Index positions of each Dict key in the flattened vector."""
    zeros = {k: np.zeros(s.shape, dtype=np.float64) for k, s in space.spaces.items()}
    out = {}
    for key in space.spaces:
        probe = {k: v.copy() for k, v in zeros.items()}
        probe[key] = probe[key] + 1.0
        flat = spaces.flatten(space, probe)
        out[key] = np.flatnonzero(flat == 1.0).astype(np.int64).tolist()
    return out


def build_extractor_kwargs(n_intruders=10, n_obstacles=5, embed_dim=64, n_heads=4,
                           intent=False, route_obs=False, probe=False, share_intent=False):
    """Keyword arguments for SetEncoderExtractor (plain, picklable data)."""
    space = competition_obs_space(n_intruders, n_obstacles, intent=intent,
                                  route_obs=route_obs, probe=probe, share_intent=share_intent)
    sl = flat_slices(space)
    total = spaces.flatten_space(space).shape[0]

    intruder_idx = [sl[k] for k in intruder_keys(intent, share_intent)]
    mask_idx = sl[MASK_KEY]
    obstacle_idx = [sl[k] for k in OBSTACLE_KEYS]
    obstacle_mask_idx = sl[OBSTACLE_MASK_KEY]
    used = ({i for idx in intruder_idx + obstacle_idx for i in idx}
            | set(mask_idx) | set(obstacle_mask_idx))

    out = {}
    if probe:
        probe_idx = [sl[k] for k in PROBE_KEYS]
        probe_mask_idx = sl[PROBE_MASK_KEY]
        used |= {i for idx in probe_idx for i in idx} | set(probe_mask_idx)
        out = dict(probe_idx=probe_idx, probe_mask_idx=probe_mask_idx)

    context_idx = [i for i in range(total) if i not in used]
    return dict(intruder_idx=intruder_idx, mask_idx=mask_idx,
                obstacle_idx=obstacle_idx, obstacle_mask_idx=obstacle_mask_idx,
                context_idx=context_idx, mode="attention",
                embed_dim=embed_dim, n_heads=n_heads, **out)


class _SetBranch(nn.Module):
    """Embed each entity, then pool with one attention query from the context."""

    def __init__(self, n_feat, embed_dim, n_heads, ctx_dim):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(n_feat, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, embed_dim), nn.ReLU(),
        )
        self.query = nn.Sequential(nn.Linear(ctx_dim, embed_dim), nn.ReLU())
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.out_dim = embed_dim

    def forward(self, entities, valid, context):
        # attention over an empty set is undefined; slot 0 holds the
        # "nothing there" padding, so use it
        empty = ~valid.any(dim=1)
        if empty.any():
            valid = valid.clone()
            valid[empty, 0] = True
        h = self.phi(entities)                                   # [B, N, E]
        q = self.query(context).unsqueeze(1)                     # [B, 1, E]
        pooled, _ = self.attn(q, h, h, key_padding_mask=~valid, need_weights=False)
        return pooled.squeeze(1)


class SetEncoderExtractor(BaseFeaturesExtractor):
    """Attention branches over intruders, obstacles and (optionally) probe
    candidates, concatenated with the context and passed through an MLP."""

    def __init__(self, observation_space, intruder_idx, mask_idx, context_idx,
                 obstacle_idx, obstacle_mask_idx, mode="attention", embed_dim=64,
                 n_heads=4, features_dim=256, probe_idx=None, probe_mask_idx=None):
        super().__init__(observation_space, features_dim)
        if mode != "attention":
            raise ValueError(f"unknown mode {mode!r}")

        self.register_buffer("intruder_idx", torch.tensor(intruder_idx, dtype=torch.long))
        self.register_buffer("mask_idx", torch.tensor(mask_idx, dtype=torch.long))
        self.register_buffer("context_idx", torch.tensor(context_idx, dtype=torch.long))
        n_ctx = self.context_idx.shape[0]

        self.intruder_branch = _SetBranch(self.intruder_idx.shape[0], embed_dim, n_heads, n_ctx)
        pooled_dim = self.intruder_branch.out_dim

        self.register_buffer("obstacle_idx", torch.tensor(obstacle_idx, dtype=torch.long))
        self.register_buffer("obstacle_mask_idx", torch.tensor(obstacle_mask_idx, dtype=torch.long))
        self.obstacle_branch = _SetBranch(self.obstacle_idx.shape[0], embed_dim, n_heads, n_ctx)
        pooled_dim += self.obstacle_branch.out_dim

        self.has_probe = probe_idx is not None
        if self.has_probe:
            self.register_buffer("probe_idx", torch.tensor(probe_idx, dtype=torch.long))
            self.register_buffer("probe_mask_idx", torch.tensor(probe_mask_idx, dtype=torch.long))
            self.probe_branch = _SetBranch(self.probe_idx.shape[0], embed_dim, n_heads, n_ctx)
            pooled_dim += self.probe_branch.out_dim

        self.head = nn.Sequential(
            nn.Linear(n_ctx + pooled_dim, features_dim), nn.ReLU(),
            nn.Linear(features_dim, features_dim), nn.ReLU(),
        )

    @staticmethod
    def _gather(obs, idx):
        """[F, N] index table -> [B, N, F] entity tensor."""
        return torch.stack([obs.index_select(1, idx[f]) for f in range(idx.shape[0])], dim=-1)

    def forward(self, obs):
        context = obs.index_select(1, self.context_idx)
        pooled = [
            self.intruder_branch(self._gather(obs, self.intruder_idx),
                                 obs.index_select(1, self.mask_idx) > 0.5, context),
            self.obstacle_branch(self._gather(obs, self.obstacle_idx),
                                 obs.index_select(1, self.obstacle_mask_idx) > 0.5, context),
        ]
        if self.has_probe:
            pooled.append(self.probe_branch(self._gather(obs, self.probe_idx),
                                            obs.index_select(1, self.probe_mask_idx) > 0.5,
                                            context))
        return self.head(torch.cat([context] + pooled, dim=-1))
