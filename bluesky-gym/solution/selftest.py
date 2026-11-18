"""Quick checks of the solution code. Run from bluesky-gym/:

    python -m solution.selftest
"""

import json
import os
import tempfile

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from gymnasium import spaces  # noqa: E402

from bluesky_gym.envs.competition_env import CompetitionEnv  # noqa: E402
from bluesky_zoo.competition.competition import CompetitionZooEnv  # noqa: E402
from solution.encoder import (SetEncoderExtractor, build_extractor_kwargs,  # noqa: E402
                              competition_obs_space)
from solution.envs import (ShapedCompetitionEnv, ShapedCompetitionZooEnv,  # noqa: E402
                           official_seed_clash)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBMISSION = os.path.join(os.path.dirname(REPO), "submission")

# the single-agent MDP of the submitted model
FINAL_SA = dict(route_mode="visibility", route_budget_frac=0.8, route_budget_hard_frac=0.95,
                route_least_penetration=True, intent=True, route_obs=True, probe=True,
                turn_guard_deg=135.0, speed_mode="anchored", speed_min_kt=200.0)


def check_scoring_layer():
    """The scored metrics come from the base envs' own code."""
    for ours, base in ((ShapedCompetitionEnv, CompetitionEnv),
                       (ShapedCompetitionZooEnv, CompetitionZooEnv)):
        for name in ("_update_metrics", "_get_info"):
            assert getattr(ours, name) is getattr(base, name), f"{ours.__name__}.{name} overridden"
    print("  _update_metrics and _get_info are inherited unchanged")


def check_trainer_kwargs():
    """Every option the trainers can set is accepted by both envs."""
    from solution.training import common_parser, env_kwargs
    args = common_parser("x").parse_args(
        "--out x --intent --route-obs --probe --share-intent --route-mode visibility "
        "--route-budget-hard-frac 0.9 --route-least-penetration --route-lateral-offset-nm 3 "
        "--speed-mode anchored --turn-guard-deg 135 --action-frequency 5 --w-obstacle -0.85 "
        "--w-miss -100 --w-intrusion-event -50 --w-restricted-event -150 --w-exit-event -150 "
        "--w-sep-barrier -1 --replay-prob 0.3".split())
    kw = env_kwargs(args)
    ShapedCompetitionEnv(**kw).close()
    ShapedCompetitionZooEnv(n_agents=10, **kw).close()
    print(f"  both envs accept all {len(kw)} trainer kwargs")


def check_spaces():
    """Env observation widths match the simulator-free space the MA trainer uses."""
    from solution.train_ma import analytic_obs_space
    expect = {(False, False): (185, 175), (True, False): (482, 470), (True, True): (532, 515)}
    for (full, share), (sa_dim, ma_dim) in expect.items():
        kw = dict(FINAL_SA, share_intent=share) if full else {}
        sa = spaces.flatten_space(ShapedCompetitionEnv(**kw).observation_space).shape[0]
        ma_env = ShapedCompetitionZooEnv(n_agents=10, **kw)
        ma = spaces.flatten_space(ma_env.observation_space(ma_env.possible_agents[0])).shape[0]
        assert ma == analytic_obs_space(10, kw).shape[0], "MA parent/worker width mismatch"
        assert (sa, ma) == (sa_dim, ma_dim), ((full, share), sa, ma)
    print("  widths: SA 185 / MA 175 plain; SA 482 / MA 515 as submitted")


def _permute(x, feat_idx, mask_idx, perm):
    out = x.clone()
    for idx in feat_idx:
        out[:, idx] = x[:, idx][:, perm]
    out[:, mask_idx] = x[:, mask_idx][:, perm]
    return out


def check_encoder():
    """Each set branch ignores slot order and masked-out entities."""
    space = spaces.flatten_space(competition_obs_space(10, 5, intent=True, route_obs=True,
                                                       probe=True))
    kw = build_extractor_kwargs(10, 5, intent=True, route_obs=True, probe=True)
    ex = SetEncoderExtractor(space, **kw).eval()
    torch.manual_seed(0)
    x = torch.rand(4, space.shape[0])
    for i in kw["mask_idx"] + kw["obstacle_mask_idx"] + kw["probe_mask_idx"]:
        x[:, i] = 1.0
    rng = np.random.default_rng(0)
    with torch.no_grad():
        base = ex(x)
        assert base.shape == (4, 256)
        for feat, mask in (("intruder_idx", "mask_idx"), ("obstacle_idx", "obstacle_mask_idx"),
                           ("probe_idx", "probe_mask_idx")):
            perm = list(rng.permutation(len(kw[mask])))
            d = (base - ex(_permute(x, kw[feat], kw[mask], perm))).abs().max().item()
            assert d < 1e-5, f"{feat} branch depends on slot order ({d:.1e})"
        x2 = x.clone()
        x2[:, kw["mask_idx"][:3]] = 0.0
        x3 = x2.clone()
        for idx in kw["intruder_idx"]:
            x3[:, idx[:3]] = 77.0
        assert (ex(x2) - ex(x3)).abs().max().item() < 1e-5, "masked intruders leak"
    print("  intruder, obstacle and probe branches: order-invariant, masked slots ignored")


def check_submitted_models():
    """The submitted models load and act in the env their _meta.json describes."""
    from stable_baselines3 import PPO
    from solution.evaluate import resolve_mdp_kw
    import gymnasium as gym
    from solution.wrappers import FlattenObs
    for kind in ("sa", "ma"):
        path = os.path.join(SUBMISSION, kind, f"ppo_{kind}.zip")
        if not os.path.exists(path):
            print(f"  {path} not found, skipped")
            continue
        kw = resolve_mdp_kw(path)
        model = PPO.load(path, device="cpu")
        if kind == "sa":
            env = gym.wrappers.FlattenObservation(ShapedCompetitionEnv(**kw))
            obs, _ = env.reset(seed=1)
        else:
            env = FlattenObs(ShapedCompetitionZooEnv(n_agents=10, **kw))
            obs, _ = env.reset(seed=1)
            obs = obs[env.agents[0]]
        assert model.observation_space.shape == obs.shape, (kind, model.observation_space, obs.shape)
        action, _ = model.predict(obs, deterministic=True)
        assert action.shape == (2,) and np.all(np.abs(action) <= 1.0 + 1e-6)
        env.close()
    print("  submitted SA and MA models load and act on their own observation")


def check_route_planner():
    """Routes keep their clearance; with the lateral offset, opposite directions
    around one obstacle pass at least 5 NM apart (without it they share a lane)."""
    from solution.route import RoutePlanner, to_latlon, to_nm
    ref = (52.0, 4.0)
    ll = lambda pts: to_latlon(np.asarray(pts, dtype=float), *ref)
    sector = ll([(100, -100), (100, 100), (-100, 100), (-100, -100)])
    box = ll([(10, -10), (10, 10), (-10, 10), (-10, -10)])
    west, east = ll([(5, -60)])[0], ll([(5, 60)])[0]

    def routes(d):
        p = RoutePlanner(clearance_nm=2.5, lateral_offset_nm=d)
        p.reset([box], sector, ref)
        out = []
        for a, b in ((west, east), (east, west)):
            wps = p.plan(a, b)
            assert len(wps), "no route around a single obstacle"
            assert p.min_clearance(np.vstack(([a], wps))) >= 2.5 - 1e-6, "route cuts the clearance"
            out.append(np.vstack((to_nm([a], *ref), to_nm(wps, *ref))))
        return out

    def gap(r1, r2):
        s = np.linspace(0, 1, 400)
        pts = np.vstack([p + np.outer(s, q - p) for p, q in zip(r1[:-1], r1[1:])])
        pts = pts[np.abs(pts[:, 1]) <= 12.0]            # alongside the obstacle
        best = np.inf
        for p, q in zip(r2[:-1], r2[1:]):
            t = np.clip((pts - p) @ (q - p) / max((q - p) @ (q - p), 1e-12), 0, 1)
            best = min(best, np.linalg.norm(pts - (p + t[:, None] * (q - p)), axis=1).min())
        return best

    g0, g3 = gap(*routes(0.0)), gap(*routes(3.0))
    assert g0 < 1.0 and g3 >= 5.0, (g0, g3)
    print(f"  opposite directions alongside the obstacle: {g0:.1f} NM apart, "
          f"{g3:.1f} NM with a 3 NM offset; clearance kept")


def check_probe_accuracy(seed=1003, k0=7):
    """The probe's predicted track matches what BlueSky then flies."""
    import bluesky as bs
    from bluesky.tools.aero import kts, vcas2tas
    from solution.probe import flat_nm, kinematic_latlon

    env = ShapedCompetitionEnv(**FINAL_SA)
    p = env.probe_obs
    prefix = np.random.default_rng(seed).uniform(-1, 1, size=(k0, 2))
    worst, pred_s, flown_s = 0.0, 0.0, 0.0
    for c in (0, p.n // 2, p.n - 1):
        env.reset(seed=seed)
        for a in prefix:
            env.step(a)
        feat = env._probe_observe(env.agent)
        i = bs.traf.id2idx(env.agent)
        h, cas = p.commands(float(bs.traf.hdg[i]), env._anchor_qdr(env.agent),
                            env.goal_heading_action.turn_guard_deg, env.speed_action.target_kt)
        lat, lon = kinematic_latlon(
            float(bs.traf.lat[i]), float(bs.traf.lon[i]), float(bs.traf.hdg[i]),
            float(bs.traf.tas[i]), h[c:c + 1], vcas2tas(cas[c:c + 1] * kts, float(bs.traf.alt[i])),
            float(bs.traf.perf.axmax[i]), env._bank(i), p.horizon)
        hold_h, hold_v = float(h[c]), float(cas[c])
        env.goal_heading_action.execute = lambda ac, a, la, lo: bs.stack.stack(f"HDG {ac} {hold_h}")
        env.speed_action.execute = lambda ac, a: bs.stack.stack(f"SPD {ac} {hold_v:.2f}")
        m0 = dict(env.metrics[env.agent])
        for k in range(p.horizon // env.action_frequency):
            _, _, te, tr, info = env.step(np.zeros(2))
            j = bs.traf.id2idx(env.agent)
            t = (k + 1) * env.action_frequency
            worst = max(worst, float(flat_nm(bs.traf.lat[j], bs.traf.lon[j], lat[0, t], lon[0, t])))
            assert not (te or tr), "test flight ended early"
        del env.goal_heading_action.execute, env.speed_action.execute
        for name, metric in (("probe_los", "intrusion_time"),
                             ("probe_obstacle", "time_in_restricted_area")):
            pred_s += float(feat[name][c]) * p.horizon
            flown_s += info[metric] - m0[metric]
    env.close()
    assert worst < 0.3, f"predicted track off by {worst:.2f} NM"
    assert abs(pred_s - flown_s) <= 0.1 * max(flown_s, 20.0), (pred_s, flown_s)
    print(f"  3 candidates flown 240 s: max position error {worst:.3f} NM; "
          f"unsafe seconds predicted {pred_s:.0f}, flown {flown_s:.0f}")


def check_action_frequency():
    """At a 5 s decision interval the per-step costs halve and a step is 5 s."""
    from solution.resume import per_decision_discount
    assert per_decision_discount(10) == (0.995, 0.95)
    g, lam = per_decision_discount(5)
    assert abs(g ** 2 - 0.995) < 1e-12 and abs(lam ** 2 - 0.95) < 1e-12
    r = {}
    for af in (10, 5):
        env = ShapedCompetitionEnv(action_frequency=af)
        env.reset(seed=4)
        env._reward_baseline = dict(env.metrics[env.agent])
        r[af] = env._get_reward(env.agent)
        if af == 5:
            t0 = env.metrics[env.agent]["flight_time"]
            _, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
            assert info["flight_time"] - t0 == 5
        env.close()
    assert r[10] != 0.0 and abs(r[5] - 0.5 * r[10]) < 1e-12, r
    print(f"  same state: {r[10]:.4f} per decision at 10 s, {r[5]:.4f} at 5 s; gamma {g:.5f}")


def check_conflict_timed_traffic(n=40):
    """Re-timed intruders stay in their speed range; off by default; training only."""
    from solution.evaluate import resolve_mdp_kw
    from solution.traffic import ConflictTimedScenarioGenerator
    gen = ConflictTimedScenarioGenerator(n_obstacles=5, n_intruders=10, agent_speed=150.0,
                                         center=(52.0, 4.0))
    rng = np.random.default_rng(5)
    lo, hi = gen.intruder_speed_range
    for _ in range(n):
        for r in gen.generate(rng).intruder_routes:
            assert lo - 1e-9 <= r.speed <= hi + 1e-9 and len(r.waypoints) >= 2
    assert 0 < gen.n_retimed < gen.n_routes

    env = ShapedCompetitionEnv()
    assert env.hard_traffic_prob == 0.0 and env._hard_gen is None
    env.close()
    env = ShapedCompetitionEnv(hard_traffic_prob=1.0, replay_seed=0)
    for k in range(4):
        env.reset(seed=3 if k == 0 else None)
    assert env.replay_stats["hard"] == 4, env.replay_stats
    env.close()
    try:
        ShapedCompetitionZooEnv(n_agents=10, hard_traffic_prob=0.5)
        raise AssertionError("the multi-agent env has no intruders to re-time")
    except ValueError:
        pass
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "ppo_sa_meta.json"), "w") as fh:
            json.dump(dict(route_mode="visibility", hard_traffic_prob=0.5, replay_prob=0.3,
                           route_lateral_offset_nm=3.0), fh)
        kw = resolve_mdp_kw(os.path.join(d, "ppo_sa.zip"))
    assert kw.get("route_lateral_offset_nm") == 3.0
    assert not any(k.startswith(("hard_traffic", "replay")) for k in kw), kw
    print(f"  {gen.n_retimed}/{gen.n_routes} intruders re-timed; off by default; "
          f"never passed to evaluation")


def check_resume():
    """Resuming keeps the update size on fewer workers and re-applies settings."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from solution.resume import (apply_hparam_overrides, resize_rollout, rollout_steps,
                                 saved_rollout_size, set_discount)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.zip")
        PPO("MlpPolicy", make_vec_env("Pendulum-v1", n_envs=2), n_steps=64, batch_size=64,
            ent_coef=0.02, device="cpu").save(path)
        assert saved_rollout_size(path) == 128
        model = PPO.load(path, env=make_vec_env("Pendulum-v1", n_envs=4), device="cpu")
        assert resize_rollout(model, rollout_steps(128, model.n_envs)) == (64, 32)
        assert apply_hparam_overrides(model, ent_coef=0.0) == {"ent_coef": (0.02, 0.0)}
        set_discount(model, 0.9975, 0.97)
        assert model.rollout_buffer.gamma == 0.9975 and model.rollout_buffer.gae_lambda == 0.97
        model.learn(total_timesteps=128)
    print("  2 -> 4 envs keeps 128 samples per update; overrides and discount re-applied")


def check_seed_guard():
    assert official_seed_clash(4, 39) and official_seed_clash(42, 1)
    assert not official_seed_clash(43, 39) and not official_seed_clash(0, 39)
    print("  training seed ranges containing 42 are refused")


CHECKS = (("scoring layer untouched", check_scoring_layer),
          ("trainer kwargs accepted", check_trainer_kwargs),
          ("observation spaces", check_spaces),
          ("set encoder", check_encoder),
          ("submitted models", check_submitted_models),
          ("route planner and lateral offset", check_route_planner),
          ("manoeuvre probe accuracy", check_probe_accuracy),
          ("decision interval", check_action_frequency),
          ("conflict-timed traffic", check_conflict_timed_traffic),
          ("resume helpers", check_resume),
          ("official seed guard", check_seed_guard))


def main():
    for name, fn in CHECKS:
        print(f"{name}:")
        fn()
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
