"""Helpers for resuming training from a checkpoint.

PPO.load restores the hyper-parameters stored in the checkpoint, so anything
that should change on a resume (entropy, epochs, KL target, learning rate,
rollout length, discount) has to be set again after loading.
"""

import json
import os
import zipfile

# read by PPO on every update, so assigning them after load is enough
OVERRIDABLE = ("ent_coef", "learning_rate", "target_kl", "n_epochs",
               "clip_range_vf", "max_grad_norm", "vf_coef")


class LinearLR:
    """Learning rate going linearly from lr0 to lr_final over the n_steps of
    this learn() call, starting at step n_start.

    On a resume SB3's progress_remaining is computed over checkpoint steps +
    new steps, so a plain `lambda p: lr0 * p` would not start at lr0.
    """

    def __init__(self, lr0, lr_final, n_start, n_steps):
        self.lr0, self.lr_final = float(lr0), float(lr_final)
        self.n_start, self.n_steps = int(n_start), max(1, int(n_steps))

    def __call__(self, progress_remaining):
        n = (1.0 - progress_remaining) * (self.n_start + self.n_steps)
        frac = min(1.0, max(0.0, (n - self.n_start) / self.n_steps))
        return self.lr0 + (self.lr_final - self.lr0) * frac


def set_lr_schedule(model, schedule):
    model.learning_rate = schedule
    model.lr_schedule = schedule


def vecnorm_candidates(resume_path, ckpt_dir):
    """Where the VecNormalize stats of a checkpoint may be, most specific first."""
    d, base = os.path.split(resume_path)
    stem = os.path.splitext(base)[0]
    return [os.path.join(d, base.replace("ck_", "ck_vecnormalize_", 1).replace(".zip", ".pkl")),
            os.path.join(d, stem + "_ck", "vecnormalize.pkl"),
            os.path.join(d, "vecnormalize.pkl"),
            os.path.join(ckpt_dir, "vecnormalize.pkl")]


def rollout_steps(rollout_size, n_envs):
    """Per-env n_steps giving rollout_size samples per PPO update."""
    return max(1, int(round(rollout_size / n_envs)))


def saved_rollout_size(path):
    """n_steps * n_envs of a saved model (read from the zip, since PPO.load
    replaces n_envs with the new env count)."""
    with zipfile.ZipFile(path) as zf:
        data = json.loads(zf.read("data").decode("utf-8"))
    return int(data["n_steps"]) * int(data["n_envs"])


def resize_rollout(model, n_steps):
    """Change the per-env rollout length of a loaded model, so a job with
    fewer workers keeps the same update size. Returns (old, new) or None."""
    old = int(model.n_steps)
    if n_steps == old:
        return None
    model.n_steps = int(n_steps)
    model.rollout_buffer = model.rollout_buffer_class(
        model.n_steps, model.observation_space, model.action_space,
        device=model.device, gamma=model.gamma, gae_lambda=model.gae_lambda,
        n_envs=model.n_envs, **model.rollout_buffer_kwargs)
    return old, int(n_steps)


def per_decision_discount(action_frequency, gamma=None, gae_lambda=None):
    """(gamma, gae_lambda) keeping the 10 s values' horizon in seconds."""
    k = action_frequency / 10.0
    return (0.995 ** k if gamma is None else gamma,
            0.95 ** k if gae_lambda is None else gae_lambda)


def set_discount(model, gamma, gae_lambda):
    """Set gamma / gae_lambda on a loaded model and its rollout buffer."""
    changed = {}
    for name, value in (("gamma", gamma), ("gae_lambda", gae_lambda)):
        old = getattr(model, name)
        if old != value:
            setattr(model, name, value)
            setattr(model.rollout_buffer, name, value)
            changed[name] = (old, value)
    return changed


def apply_hparam_overrides(model, **overrides):
    """Set the given hyper-parameters on a loaded model; None means keep."""
    changed = {}
    for name, value in overrides.items():
        if value is None:
            continue
        if name not in OVERRIDABLE:
            raise ValueError(f"{name} cannot be changed after load")
        old = getattr(model, name)
        if old == value:
            continue
        setattr(model, name, value)
        if name == "learning_rate":
            model.lr_schedule = lambda _progress, v=float(value): v
        changed[name] = (old, value)
    return changed
