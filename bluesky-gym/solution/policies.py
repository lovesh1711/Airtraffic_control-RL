"""Policy tweaks for the single-agent track.

With an unbounded Gaussian mean the SA policy drifted to |mean| >> 1: the env
clips every sample to +-1, so the entropy bonus pushed log_std up with nothing
pushing back, and the policy became a bang-bang controller. A tanh on the
mean (only the mean, so the log-density stays exact) fixes that; ClampLogStd
is an extra hard cap on the std.
"""

import numpy as np
import torch as th
from torch import nn

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.policies import ActorCriticPolicy


class BoundedMeanActorCriticPolicy(ActorCriticPolicy):
    """ActorCriticPolicy with the Gaussian mean squashed into [-1, 1]."""

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        if not isinstance(self.action_dist, DiagGaussianDistribution):
            raise TypeError("BoundedMeanActorCriticPolicy needs a diagonal Gaussian, got "
                            f"{type(self.action_dist).__name__}")
        # wrapping keeps the same parameters, so the optimizer is unaffected
        self.action_net = nn.Sequential(self.action_net, nn.Tanh())


class ClampLogStd(BaseCallback):
    """Keep log_std <= log(max_std), after every update and at the end."""

    def __init__(self, max_std=1.0, verbose=0):
        super().__init__(verbose)
        self.max_std = float(max_std)
        self.max_log_std = float(np.log(self.max_std))
        self.n_clamped = 0

    def _clamp(self):
        log_std = getattr(self.model.policy, "log_std", None)
        if log_std is None:
            return
        with th.no_grad():
            if bool((log_std > self.max_log_std).any()):
                self.n_clamped += 1
                if self.n_clamped == 1:
                    print(f"log_std clamp engaged at {self.num_timesteps} steps "
                          f"(std capped at {self.max_std})")
            log_std.clamp_(max=self.max_log_std)

    def _on_rollout_start(self):
        self._clamp()

    def _on_training_end(self):
        self._clamp()

    def _on_step(self):
        return True


def describe_policy(model):
    policy = model.policy
    log_std = getattr(policy, "log_std", None)
    std = ("n/a" if log_std is None
           else np.array2string(th.exp(log_std.detach()).cpu().numpy(), precision=3))
    bounded = isinstance(policy, BoundedMeanActorCriticPolicy)
    return f"policy: {type(policy).__name__} (mean bounded by tanh: {bounded}), std = {std}"
