"""The competition environments with our observation, action and reward.

The scoring layer (_update_metrics / _get_info) is inherited unchanged, so the
reported metrics are the official ones.
"""

from gymnasium import spaces

from bluesky_gym.envs.competition_env import CompetitionEnv
from bluesky_zoo.competition.competition import CompetitionZooEnv

from solution.actions import GoalReferencedActionMixin
from solution.geometry import clear as clear_cpa_cache
from solution.observations import (RiskIntruderObservation, ThreatObstacleObservation,
                                   ThreatBoundaryObservation, RouteObservation)
from solution.probe import ManoeuvreProbeMixin
from solution.replay import ScenarioReplayMixin
from solution.shaping import ShapedRewardMixin

# The official evaluation seeds the env with 42 once. Training worker i is
# seeded seed + i, and a worker seeded 42 would train on the test scenarios.
OFFICIAL_EVAL_SEED = 42


def official_seed_clash(seed, n_workers):
    return seed <= OFFICIAL_EVAL_SEED < seed + n_workers


SHAPING_KEYS = ("w_progress", "w_separation", "w_obstacle", "w_boundary", "w_step",
                "w_action", "w_smooth", "warn_nm", "tcpa_horizon", "obstacle_clearance_nm",
                "boundary_margin_nm", "w_miss", "w_intrusion_event", "w_restricted_event",
                "w_exit_event", "w_sep_barrier", "sep_barrier_nm", "separation_mode")
OBS_KEYS = ("intent",)
ACTION_KEYS = ("route_mode", "route_clearance_nm", "route_switch_nm", "route_budget_frac",
               "route_budget_hard_frac", "route_least_penetration", "route_lateral_offset_nm",
               "speed_mode", "speed_min_kt", "speed_max_kt", "turn_guard_deg")
# training only, never passed at evaluation
REPLAY_KEYS = ("replay_prob", "replay_size", "replay_temperature", "replay_miss_penalty",
               "replay_seed", "hard_traffic_prob", "hard_traffic_frac", "hard_traffic_jitter_s")


def _split_kwargs(kwargs):
    pick = lambda keys: {k: kwargs.pop(k) for k in list(kwargs) if k in keys}
    return pick(SHAPING_KEYS), pick(OBS_KEYS), pick(ACTION_KEYS), pick(REPLAY_KEYS), kwargs


class _SolutionMixin:
    """Observation setup and extras shared by both envs."""

    def _setup(self, n_others, route_obs, probe, share_intent, obs_kw, shaping, action_kw,
               replay_kw):
        self.intruder_obs = RiskIntruderObservation(
            n=n_others, closest_fn=self._plan_closest,
            share=share_intent, share_fn=self._intent_row, **obs_kw)
        self.obstacle_obs = ThreatObstacleObservation(n=self.n_obstacles,
                                                      polygons=self._obstacle_polygons)
        self.sector_obs = ThreatBoundaryObservation(n=12, distance_norm=170.0)
        self.route_obs = RouteObservation() if route_obs else None
        self._configure_shaping(**shaping)
        self._configure_actions(**action_kw)
        self._configure_replay(**replay_kw)
        # after the actions: the probe's candidates are the action's commands
        self._configure_probe(probe=probe, share_intent=share_intent)
        return spaces.Dict({
            **self.waypoint_obs.space(),
            **self.airspeed_obs.space(),
            **self.intruder_obs.space(),
            **self.obstacle_obs.space(),
            **self.sector_obs.space(),
            **(self.route_obs.space() if self.route_obs else {}),
            **(self.probe_obs.space() if self.probe_obs is not None else {}),
        })

    def _obstacle_polygons(self):
        # read live: the base reset builds its first observation before ours runs
        return [o.vertices for o in self.scenario.obstacles]

    def _sector_vertices(self):
        return self.scenario.sector

    def _get_obs(self, ac_id):
        obs = super()._get_obs(ac_id)
        if self.route_obs is not None:
            anchor, nxt, remaining = self._route_progress(ac_id)
            obs.update(self.route_obs.observe(
                ac_id, anchor, nxt, remaining, self._time_slack(ac_id, remaining)))
        if getattr(self, "probe_obs", None) is not None:
            obs.update(self._probe_observe(ac_id))
        return obs

    def _needs_fresh_obs(self):
        return (self._planner is not None or self.intruder_obs.intent or self.route_obs
                or self.probe_obs is not None or self.share_intent)


class ShapedCompetitionEnv(_SolutionMixin, ScenarioReplayMixin, ManoeuvreProbeMixin,
                           GoalReferencedActionMixin, ShapedRewardMixin, CompetitionEnv):
    """Single-agent env. reach_reward is raised from 1 so arriving is worth
    something next to the safety penalties."""

    def __init__(self, reach_reward=20.0, route_obs=False, probe=False, share_intent=False,
                 **kwargs):
        shaping, obs_kw, action_kw, replay_kw, env_kw = _split_kwargs(kwargs)
        super().__init__(reach_reward=reach_reward, **env_kw)
        self.observation_space = self._setup(self.n_intruders, route_obs, probe, share_intent,
                                             obs_kw, shaping, action_kw, replay_kw)

    def _goal_latlon(self, ac_id):
        return self.goal_lat, self.goal_lon

    def _baseline_of(self, ac_id):
        return self._reward_baseline

    def reset(self, seed=None, options=None):
        # BlueSky does not rewind its clock on reset, so clear the CPA cache
        # before the base reset builds the first observation
        clear_cpa_cache()
        self._replay_before_reset()
        observation, info = super().reset(seed=seed, options=options)
        self._replay_after_reset()
        self._reset_shaping()
        self._reset_routes()
        if self._needs_fresh_obs():
            # the base reset observed before this episode's routes existed
            observation = self._get_obs(self.agent)
        return observation, info

    def step(self, action):
        out = super().step(action)
        if out[2] or out[3]:
            self._replay_episode_done([self.metrics[self.agent]])
        return out


class ShapedCompetitionZooEnv(_SolutionMixin, ScenarioReplayMixin, ManoeuvreProbeMixin,
                              GoalReferencedActionMixin, ShapedRewardMixin, CompetitionZooEnv):
    """Multi-agent env (one shared policy, each aircraft sees the other nine)."""

    def __init__(self, reach_reward=20.0, route_obs=False, probe=False, share_intent=False,
                 **kwargs):
        shaping, obs_kw, action_kw, replay_kw, env_kw = _split_kwargs(kwargs)
        super().__init__(reach_reward=reach_reward, **env_kw)
        space = self._setup(self.n_agents - 1, route_obs, probe, share_intent,
                            obs_kw, shaping, action_kw, replay_kw)
        self.observation_spaces = {a: space for a in self.possible_agents}

    def _goal_latlon(self, ac_id):
        return self._goal[ac_id]

    def _baseline_of(self, ac_id):
        return self._reward_baseline[ac_id]

    def _route_agents(self):
        return list(self.agents)

    def reset(self, seed=None, options=None):
        clear_cpa_cache()
        self._replay_before_reset()
        observations, infos = super().reset(seed=seed, options=options)
        self._replay_after_reset()
        self._reset_shaping()
        self._reset_routes()
        if self._needs_fresh_obs():
            observations = {a: self._get_obs(a) for a in observations}
        return observations, infos

    def step(self, actions):
        out = super().step(actions)
        if not self.agents:
            self._replay_episode_done([self.metrics[a] for a in self.possible_agents
                                       if a in self.metrics])
        return out
