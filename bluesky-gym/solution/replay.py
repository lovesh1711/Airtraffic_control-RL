"""Training-only scenario sources: prioritised replay and conflict-timed traffic.

Replay (after Prioritized Level Replay, Jiang et al. 2021): each env keeps a
buffer of scenarios it flew badly, scored by unsafe seconds (+ a penalty for a
missed goal), and with probability replay_prob re-flies one of them, picked by
rank. A scenario flown cleanly again is dropped.

Conflict-timed traffic: with probability hard_traffic_prob a fresh scenario
comes from solution.traffic instead, with intruders re-timed to meet the agent.

The rules allow custom training scenarios as long as evaluation uses the
original ones; neither option is ever set on the evaluation path.
"""

import numpy as np


class ScenarioReplayMixin:
    """Mix in before the env class (uses its _fixed_scenario hook)."""

    def _configure_replay(self, replay_prob=0.0, replay_size=256,
                          replay_temperature=0.3, replay_miss_penalty=500.0,
                          replay_seed=None, hard_traffic_prob=0.0,
                          hard_traffic_frac=1.0, hard_traffic_jitter_s=60.0):
        if not 0.0 <= replay_prob < 1.0:
            raise ValueError(f"replay_prob must be in [0, 1), got {replay_prob}")
        if not 0.0 <= hard_traffic_prob <= 1.0:
            raise ValueError(f"hard_traffic_prob must be in [0, 1], got {hard_traffic_prob}")
        self.hard_traffic_prob = float(hard_traffic_prob)
        self._hard_gen = None
        if self.hard_traffic_prob > 0.0:
            if not getattr(self, "n_intruders", 0):
                raise ValueError("hard_traffic_prob re-times scripted intruders; "
                                 "this environment has none (single-agent only)")
            from solution.traffic import ConflictTimedScenarioGenerator
            self._hard_gen = ConflictTimedScenarioGenerator(
                retime_frac=hard_traffic_frac, jitter_s=hard_traffic_jitter_s,
                n_obstacles=self.n_obstacles, n_intruders=self.n_intruders,
                agent_speed=self.ac_spd, center=self.center)
        self.replay_prob = float(replay_prob)
        self.replay_size = int(replay_size)
        self.replay_temperature = float(replay_temperature)
        self.replay_miss_penalty = float(replay_miss_penalty)
        # without a seed, the sampler is seeded from the env RNG state after
        # the first reset (read, not drawn from, so scenarios are unchanged)
        self._replay_rng = (None if replay_seed is None
                            else np.random.default_rng(replay_seed))
        self._replay_buf = []           # [scenario, score]
        self._replay_current = None
        self._replay_user_fixed = None
        self._replay_open = False
        self.replay_stats = dict(fresh=0, replayed=0, stored=0, evicted=0, hard=0)

    def _replay_pick(self):
        scores = np.array([s for _, s in self._replay_buf])
        ranks = np.empty(len(scores))
        ranks[np.argsort(-scores, kind="stable")] = np.arange(1, len(scores) + 1)
        w = (1.0 / ranks) ** (1.0 / self.replay_temperature)
        return int(self._replay_rng.choice(len(scores), p=w / w.sum()))

    def _replay_before_reset(self):
        self._replay_current = None
        if self.replay_prob <= 0.0 and self.hard_traffic_prob <= 0.0:
            return
        # a scenario fixed by the caller wins
        if self._replay_user_fixed is None:
            self._replay_user_fixed = self._fixed_scenario is not None
        if self._replay_user_fixed:
            return
        if (self.replay_prob > 0.0 and self._replay_buf
                and self._replay_rng.random() < self.replay_prob):
            self._replay_current = self._replay_pick()
            self._fixed_scenario = self._replay_buf[self._replay_current][0]
            self.replay_stats["replayed"] += 1
        else:
            self._fixed_scenario = self._hard_draw()
            self.replay_stats["fresh"] += 1
        self._replay_open = self.replay_prob > 0.0

    def _hard_draw(self):
        """A conflict-timed scenario with probability hard_traffic_prob, else None."""
        if self._hard_gen is None or self._replay_rng is None:
            return None
        if self._replay_rng.random() >= self.hard_traffic_prob:
            return None
        self.replay_stats["hard"] += 1
        return self._hard_gen.generate(self._replay_rng)

    def _replay_after_reset(self):
        if self.replay_prob <= 0.0 and self.hard_traffic_prob <= 0.0:
            return
        if self._replay_rng is None:
            env_rng = getattr(self, "_np_random", None)
            state = (env_rng.bit_generator.state["state"]["state"]
                     if env_rng is not None else None)
            self._replay_rng = np.random.default_rng(
                None if state is None else int(state) % (2 ** 63))
        if not self._replay_user_fixed:
            self._fixed_scenario = None

    def _replay_score(self, metrics_list):
        harm = 0.0
        for m in metrics_list:
            harm += (m["intrusion_time"] + m["time_in_restricted_area"]
                     + m["time_outside_sector"])
            if not m["waypoint_reached"]:
                harm += self.replay_miss_penalty
        return harm / max(len(metrics_list), 1)

    def _replay_episode_done(self, metrics_list):
        if not self._replay_open:
            return
        self._replay_open = False
        score = self._replay_score(metrics_list)
        if self._replay_current is not None:
            # re-score; drop it once it is flown cleanly
            if score <= 0.0:
                self._replay_buf.pop(self._replay_current)
                self.replay_stats["evicted"] += 1
            else:
                self._replay_buf[self._replay_current][1] = score
            self._replay_current = None
            return
        if score <= 0.0:
            return
        if len(self._replay_buf) < self.replay_size:
            self._replay_buf.append([self.scenario, score])
            self.replay_stats["stored"] += 1
            return
        worst = int(np.argmin([s for _, s in self._replay_buf]))
        if score > self._replay_buf[worst][1]:
            self._replay_buf[worst] = [self.scenario, score]
            self.replay_stats["stored"] += 1
            self.replay_stats["evicted"] += 1


try:
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError:
    BaseCallback = object


class ReplayStatsCallback(BaseCallback):
    """Log replay/* rows into the SB3 training log."""

    def __init__(self, every=10, verbose=0):
        super().__init__(verbose)
        self.every = int(every)
        self._n = 0

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        self._n += 1
        if self._n % self.every:
            return
        try:
            stats = self.training_env.get_attr("replay_stats")
            sizes = self.training_env.get_attr("_replay_buf")
        except Exception:               # e.g. a vec env without get_attr
            return
        fresh = sum(s["fresh"] for s in stats)
        replayed = sum(s["replayed"] for s in stats)
        self.logger.record("replay/replayed_frac", replayed / max(fresh + replayed, 1))
        hard = sum(s.get("hard", 0) for s in stats)
        if hard:
            self.logger.record("replay/hard_frac", hard / max(fresh, 1))
        self.logger.record("replay/buffer_mean", float(np.mean([len(b) for b in sizes])))
        scores = [sc for b in sizes for _, sc in b]
        if scores:
            self.logger.record("replay/score_mean", float(np.mean(scores)))
