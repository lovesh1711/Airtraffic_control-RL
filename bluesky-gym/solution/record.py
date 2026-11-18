"""Scenario GIFs of a policy, for the submission video.

The organisers' scripts/record_competition_gifs.py builds the stock envs, so it
cannot run our policy; this mirrors it on our envs and reuses its save_gif.

    python -m solution.record --env sa --model ../submission/sa/ppo_sa --seeds 9 8 13 2 33 --out reel_sa.gif
    python -m solution.record --env sa --model ../submission/sa/ppo_sa --seeds 9 8 13 2 33         --render-size 640 --caption --compare-neutral --size 900 --out reel_sa_vs_neutral.gif

Without --model it records the do-nothing policy.
"""

import argparse
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np  # noqa: E402
import gymnasium as gym  # noqa: E402

from scripts.record_competition_gifs import save_gif  # noqa: E402
from solution.envs import ShapedCompetitionEnv, ShapedCompetitionZooEnv  # noqa: E402
from solution.wrappers import FlattenObs  # noqa: E402

COUNTERS = ("intrusion_time", "time_in_restricted_area", "time_outside_sector")


def load_policy(model_path):
    """act(obs) -> action, or None for the do-nothing policy (flies straight at the goal)."""
    if model_path is None:
        return None
    from stable_baselines3 import PPO
    model = PPO.load(model_path, device="cpu")
    return lambda obs: model.predict(obs, deterministic=True)[0]


def mdp_kwargs(model_path):
    """Env kwargs the model was trained with (from its _meta.json)."""
    from solution.evaluate import resolve_mdp_kw
    kw = resolve_mdp_kw(model_path)
    print("recording under MDP:", kw, flush=True)
    return kw


def _set_render_size(raw, size):
    """Draw on a size x size canvas instead of 512 (call before reset)."""
    if not size:
        return
    from core.rendering import PygameCanvas
    raw.window_width = raw.window_height = int(size)
    raw.window_size = (int(size), int(size))
    raw.pygame_canvas = PygameCanvas(int(size), int(size), mode=raw.render_mode)


def _stats(infos, n_agents):
    """Running safety counters summed over the aircraft seen so far."""
    infos = list(infos)
    s = {k: float(sum(i.get(k, 0.0) for i in infos)) for k in COUNTERS}
    s["t"] = max((float(i.get("flight_time", 0.0)) for i in infos), default=0.0)
    s["at_goal"] = int(sum(float(i.get("waypoint_reached", 0.0)) >= 1 for i in infos))
    s["n"] = n_agents
    return s


def _frame_stride(raw_env):
    """Decisions per frame, so there is one frame per 10 simulated seconds."""
    return max(1, round(10 / raw_env.action_frequency))


def rollout_single_agent(seed, act=None, max_steps=400, mdp=None,
                         render_size=None, with_stats=False):
    env = gym.wrappers.FlattenObservation(
        ShapedCompetitionEnv(render_mode="rgb_array", **(mdp or {})))
    _set_render_size(env.unwrapped, render_size)
    stride = _frame_stride(env.unwrapped)
    frames, stats = [], []
    obs, _ = env.reset(seed=seed)
    frames.append(env.render())
    stats.append(_stats([], 1))
    terminated = truncated = False
    steps = 0
    while not (terminated or truncated) and steps < max_steps * stride:
        action = np.zeros(2, dtype=np.float32) if act is None else act(obs)
        obs, _, terminated, truncated, info = env.step(action)
        steps += 1
        if steps % stride == 0 or terminated or truncated:
            frames.append(env.render())
            stats.append(_stats([info], 1))
    env.close()
    return (frames, stats) if with_stats else frames


def rollout_multi_agent(seed, n_agents=10, act=None, max_steps=400, mdp=None,
                        render_size=None, with_stats=False):
    env = FlattenObs(ShapedCompetitionZooEnv(render_mode="rgb_array",
                                             n_agents=n_agents, **(mdp or {})))
    _set_render_size(env.env, render_size)
    stride = _frame_stride(env.env)
    frames, stats, last = [], [], {}
    observations, _ = env.reset(seed=seed)
    frames.append(env.render())
    stats.append(_stats([], n_agents))
    steps = 0
    while env.agents and steps < max_steps * stride:
        if act is None:
            actions = {a: np.zeros(2, dtype=np.float32) for a in env.agents}
        else:
            actions = {a: act(observations[a]) for a in env.agents}
        observations, _, _, _, infos = env.step(actions)
        last.update(infos)
        steps += 1
        if steps % stride == 0 or not env.agents:
            frames.append(env.render())
            stats.append(_stats(last.values(), n_agents))
    env.close()
    return (frames, stats) if with_stats else frames


def _font(px):
    from PIL import ImageFont
    for name in ("DejaVuSans.ttf", "arial.ttf", "Arial.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, px)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=px)
    except TypeError:                       # Pillow < 10.1
        return ImageFont.load_default()


def _panel(frame, title, st, scenario):
    """Frame with a two-line caption bar on top."""
    from PIL import Image, ImageDraw
    img = Image.fromarray(frame)
    W, H = img.size
    fs = max(11, W // 34)
    big, small = _font(int(fs * 1.15)), _font(fs)
    bar = int(fs * 3.4)
    out = Image.new("RGB", (W, H + bar), (22, 28, 38))
    out.paste(img, (0, bar))
    d = ImageDraw.Draw(out)
    pad = fs // 2
    right = f"{scenario}   t = {st['t']:4.0f} s"
    # shrink the title until it fits next to the clock
    room = W - 3 * pad - d.textlength(right, font=small)
    px = int(fs * 1.15)
    while px > int(fs * 0.75) and d.textlength(title, font=big) > room:
        px -= 1
        big = _font(px)
    d.text((pad, pad // 2), title, font=big, fill=(240, 240, 240))
    d.text((W - pad - d.textlength(right, font=small), pad // 2 + 2), right,
           font=small, fill=(170, 180, 195))
    x, y = pad, pad // 2 + int(fs * 1.6)
    parts = [("intrusion", st["intrusion_time"]), ("restricted", st["time_in_restricted_area"]),
             ("outside", st["time_outside_sector"])]
    for name, v in parts:
        txt = f"{name} {v:.0f} s"
        d.text((x, y), txt, font=small, fill=(235, 90, 80) if v > 0 else (110, 210, 120))
        x += d.textlength(txt, font=small) + fs
    goal = (f"at goal {st['at_goal']}/{st['n']}" if st["n"] > 1
            else ("goal reached" if st["at_goal"] else "en route"))
    d.text((x, y), goal, font=small, fill=(200, 200, 200))
    return out


def _hstack(images, gap=6):
    from PIL import Image
    W = sum(i.width for i in images) + gap * (len(images) - 1)
    H = max(i.height for i in images)
    out = Image.new("RGB", (W, H), (10, 12, 16))
    x = 0
    for im in images:
        out.paste(im, (x, 0))
        x += im.width + gap
    return out


def _save_images(images, out_path, width=None, max_frames=0, duration=80):
    """GIF writer for the captioned / side-by-side reels."""
    from PIL import Image
    if max_frames and len(images) > max_frames:
        images = images[::int(np.ceil(len(images) / max_frames))]
    if width and images[0].width != width:
        h = round(images[0].height * width / images[0].width)
        images = [im.resize((width, h), Image.LANCZOS) for im in images]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    images[0].save(out_path, save_all=True, append_images=images[1:],
                   loop=0, duration=duration, optimize=True)
    mb = os.path.getsize(out_path) / 1e6
    print(f"wrote {out_path}  ({len(images)} frames, {images[0].width}x{images[0].height}, {mb:.1f} MB)")


def record(env_kind, seeds, out_path, n_agents=10, model_path=None, render_size=None,
           caption=False, compare_neutral=False, hold=12, max_steps=400, **gif_kwargs):
    act = load_policy(model_path)
    mdp = mdp_kwargs(model_path)

    def roll(seed, act, mdp):
        if env_kind == "sa":
            return rollout_single_agent(seed, act=act, mdp=mdp, max_steps=max_steps,
                                        render_size=render_size, with_stats=True)
        return rollout_multi_agent(seed, n_agents=n_agents, act=act, mdp=mdp,
                                   max_steps=max_steps, render_size=render_size,
                                   with_stats=True)

    if not (caption or compare_neutral or render_size):
        # the organisers' plain format
        frames = []
        for seed in seeds:
            frames += roll(seed, act, mdp)[0]
            print(f"  recorded seed {seed} ({len(frames)} frames so far)", flush=True)
        gif_kwargs["size"] = gif_kwargs.get("size") or 320
        save_gif(frames, out_path, **gif_kwargs)
        return

    who = "Trained RL policy" if model_path else "Neutral policy"
    track = "single-agent" if env_kind == "sa" else f"multi-agent, {n_agents} aircraft"
    images = []
    for k, seed in enumerate(seeds, 1):
        scen = f"scenario {k}/{len(seeds)}"
        f_t, s_t = roll(seed, act, mdp)
        if compare_neutral:
            f_n, s_n = roll(seed, None, {})
            L = max(len(f_t), len(f_n))
            pad = lambda xs: xs + [xs[-1]] * (L - len(xs))
            f_t, s_t, f_n, s_n = pad(f_t), pad(s_t), pad(f_n), pad(s_n)
            seq = [_hstack([_panel(a, f"Neutral (fly straight) - {track}", sa, scen),
                            _panel(b, f"{who} - {track}", sb, scen)])
                   for a, sa, b, sb in zip(f_n, s_n, f_t, s_t)]
        else:
            seq = [_panel(a, f"{who} - {track}", sa, scen) for a, sa in zip(f_t, s_t)]
        images += seq + [seq[-1]] * hold
        print(f"  recorded seed {seed} ({len(images)} frames so far)", flush=True)
    _save_images(images, out_path, width=gif_kwargs.get("size"),
                 max_frames=gif_kwargs.get("max_frames", 0),
                 duration=gif_kwargs.get("duration", 80))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", choices=["sa", "ma"], default="sa")
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5],
                   help="scenario seeds, concatenated into one GIF")
    p.add_argument("--n-agents", type=int, default=10)
    p.add_argument("--model", default=None, help="trained model; omit for do-nothing")
    p.add_argument("--out", required=True)
    p.add_argument("--size", type=int, default=None,
                   help="output width in px (default: 320 for the plain reel, "
                        "the native render width otherwise)")
    p.add_argument("--max-frames", type=int, default=None,
                   help="frames kept after subsampling (default: 400 for the "
                        "plain reel, all frames otherwise)")
    p.add_argument("--duration", type=int, default=80, help="ms per frame")
    p.add_argument("--render-size", type=int, default=None,
                   help="draw the env canvas at this many px (env default 512)")
    p.add_argument("--caption", action="store_true",
                   help="caption bar: policy, scenario, time, running safety counters")
    p.add_argument("--compare-neutral", action="store_true",
                   help="neutral policy side by side with the model, same scenarios")
    p.add_argument("--max-steps", type=int, default=400,
                   help="env steps per scenario before the recording is cut")
    args = p.parse_args()

    plain = not (args.caption or args.compare_neutral or args.render_size)
    max_frames = args.max_frames if args.max_frames is not None else (400 if plain else 0)
    record(args.env, args.seeds, args.out, n_agents=args.n_agents,
           model_path=args.model, render_size=args.render_size, caption=args.caption,
           compare_neutral=args.compare_neutral, max_steps=args.max_steps,
           size=args.size, max_frames=max_frames, duration=args.duration)


if __name__ == "__main__":
    main()
