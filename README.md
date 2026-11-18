# BlueSky-Gym ATC competition — PPO with route-anchored actions and a manoeuvre probe

Entry for the AI4REALNET / TU Delft BlueSky-Gym air traffic control competition,
both tracks:

* **Single-agent (SA):** one controlled aircraft among 10 scripted intruders.
* **Multi-agent (MA):** 10 controlled aircraft sharing one policy.

## Results (official protocol: seed 42, first 1000 episodes)

| Track | Policy | Intrusion [s] | Restricted [s] | Outside [s] | Sum [s] | Goal | Clean flights | Flight time [s] |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SA | do nothing | 96.0 | 130.0 | 0.0 | 226.0 | 100% | 9.9% | 852 |
| SA | **ours** | **12.4** | **1.8** | **0.1** | **14.3** | **100%** | **76.8%** | 1153 |
| MA | do nothing | 64.8 | 121.7 | 0.0 | 186.4 | 100% | 17.9% | 848 |
| MA | **ours** | **2.9** | **2.1** | **0.0** | **5.0** | **100%** | **91.5%** | 1066 |

Seconds per flight (MA: per aircraft, over 10 000 aircraft). A clean flight
reaches its goal with no intrusion, restricted-area entry or sector exit.
Per-episode CSVs: [`results/official/final_2545470/`](results/official/final_2545470/).

## Approach

* **PPO** (Stable-Baselines3), one shared policy for all aircraft in MA.
* **Set encoder:** intruders, obstacles and probe candidates are each pooled
  by attention, so slot order does not matter and the same network serves 10
  intruders or 9 neighbours (`solution/encoder.py`).
* **Observation:** closest-point-of-approach features, also along both
  aircraft's planned paths; exact obstacle polygons with time to entry; sector
  margin and time to exit; route progress and time slack; in MA, what each
  neighbour is commanding (`solution/observations.py`).
* **Manoeuvre probe:** before every decision, 27 candidate commands (9 heading
  offsets x 3 speeds) are flown 240 s ahead with BlueSky's own kinematics
  against the predicted traffic, and the policy sees the predicted closest
  approach, intrusion, restricted-area and sector-exit time of each
  (`solution/probe.py`). It is an input; the policy still chooses.
* **Action:** heading as an offset (±45°) from the bearing to the next waypoint
  of a visibility-graph route around the obstacles, plus a target speed
  (200–317 kt). Doing nothing therefore follows a clear path. If no clear route
  fits the time limit, the planner takes a longer one, then the one least
  inside obstacles (`solution/actions.py`, `solution/route.py`).
* **Lateral offset (MA):** routes around the same obstacle share its corners,
  which put opposite directions head-on there. Every route is flown to the
  right of the shared path, as oceanic ATC does, so they keep separate lanes.
* **Reward:** progress, anticipatory separation / obstacle / boundary costs, a
  steep barrier just outside 5 NM, one-off charges when a violation starts and
  a timeout penalty (`solution/shaping.py`). It is only a training signal; the
  scored metrics come from the environment unchanged.
* **Training:** prioritised replay of failed scenarios; in the last SA stage,
  half the fresh scenarios have intruders re-timed to meet the agent
  (`solution/traffic.py`, training only). Models were selected on a separate
  validation stream (seed 7777), never on seed 42.

## Setup

Python 3.12, CPU only.

```bash
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
cd bluesky-gym
python -m solution.selftest      # ~5 min, ends with "ALL CHECKS PASSED"
```

Run everything from `bluesky-gym/`, so `bluesky_gym`, `bluesky_zoo`, `core`
and `solution` import from the working tree.

## Reproduce the official numbers

`solution/evaluate.py` runs the organisers' `scripts/evaluate_competition.py`
with only its two edit points (`make_env`, `load_policy`) replaced by our
environments and policies. Each model's environment settings are read from the
`*_meta.json` next to it.

```bash
cd bluesky-gym
python -m solution.evaluate --env sa --episodes 1000 --model ../submission/sa/ppo_sa --out sa.csv
python -m solution.evaluate --env ma --episodes 1000 --model ../submission/ma/ppo_ma --out ma.csv
python -m solution.evaluate --env sa --episodes 1000 --out neutral_sa.csv     # do-nothing baseline
```

The protocol is sequential: about 1 h for SA and 5–6 h for MA. The same
episodes can be flown in parallel for a quicker check:

```bash
python -m solution.validate --kind sa --episodes 1000 --eval-seed 42 --workers 8 --out sa42.json ../submission/sa/ppo_sa.zip
python -m solution.validate --kind ma --episodes 1000 --eval-seed 42 --workers 8 --out ma42.json ../submission/ma/ppo_ma.zip
```

Results can differ slightly between machines through floating-point
differences: in most SA episodes the safety metrics match exactly, while in MA
ten interacting aircraft amplify the differences (about ±1 s on the mean).

## Videos

The five scenarios per track are chosen by a fixed rule: of scenario seeds
1–40, the five where our policy flies clean and the do-nothing policy is worst.

```bash
python -m solution.pick_video_seeds --env sa --model ../submission/sa/ppo_sa --out seeds_sa.json
python -m solution.record --env sa --model ../submission/sa/ppo_sa --seeds 9 8 13 2 33 --out reel_sa.gif
python -m solution.record --env sa --model ../submission/sa/ppo_sa --seeds 9 8 13 2 33 \
    --render-size 640 --caption --compare-neutral --size 900 --max-frames 500 --out reel_sa_vs_neutral.gif
```

The submitted reels are in [`results/videos/video_2549833/`](results/videos/video_2549833/)
(SA seeds 9 8 13 2 33, MA seeds 2 33 19 13 40).

## Training

The models were trained on a PBS cluster in stages, each resuming the previous
one, with the jobs in [`cluster/`](cluster/) (environment: `cluster/setup.sh`).
Submit from the repository root; `<id>` is the job id of the previous stage,
whose output lands in `results/<track>_<id>/`. Times are for 20 cores.

```bash
Q="-l select=1:ncpus=20"
# single-agent: 15M fresh, 2 x 15M with fallback routing (lr 3e-4 -> 1.5e-4 -> 0),
# 15M with conflict-timed traffic (~12 h each)
qsub $Q -v TRACK=sa,STAGE=1,TIMESTEPS=15000000,SEED=170 cluster/train.pbs
qsub $Q -v TRACK=sa,STAGE=2,TIMESTEPS=15000000,LR_FINAL=0.00015,SEED=180,RESUME=results/sa_<id>/ppo_sa.zip cluster/train.pbs
qsub $Q -v TRACK=sa,STAGE=2,TIMESTEPS=15000000,LR=0.00015,LR_FINAL=0,SEED=181,RESUME=results/sa_<id>/ppo_sa.zip cluster/train.pbs
qsub $Q -v TRACK=sa,STAGE=2,TIMESTEPS=15000000,LR=0.0001,LR_FINAL=0,HARD_TRAFFIC_PROB=0.5,SEED=200,RESUME=results/sa_<id>/ppo_sa.zip cluster/train.pbs

# multi-agent: 20M fresh (~14 h), then 3 x 20M (~8 h each), the last with the lateral offset
qsub $Q -l walltime=23:30:00 -v TRACK=ma,STAGE=1,TIMESTEPS=20000000,SEED=121 cluster/train.pbs
qsub $Q -v TRACK=ma,STAGE=2,TIMESTEPS=20000000,LR_FINAL=0,SEED=184,RESUME=results/ma_<id>/ppo_ma.zip cluster/train.pbs
qsub $Q -v TRACK=ma,STAGE=2,TIMESTEPS=20000000,LR_FINAL=0,SEED=190,RESUME=results/ma_<id>/ppo_ma.zip cluster/train.pbs
qsub $Q -v TRACK=ma,STAGE=2,TIMESTEPS=20000000,LR_FINAL=0,ROUTE_LATERAL_OFFSET_NM=3,SEED=202,RESUME=results/ma_<id>/ppo_ma.zip cluster/train.pbs
```

`cluster/train.pbs` lists the full flags; the same training runs without PBS
as `python -m solution.train_sa ...` / `python -m solution.train_ma ...`
(`--help` for the options). Training seeds keep every rollout worker off 42,
the test stream; the trainers refuse a clashing seed. Rank candidates on the
validation stream with `cluster/validate.pbs`, then run the chosen ones with
`cluster/evaluate.pbs`. Training logs of every stage are in
[`results/training/`](results/training/).

## Repository layout

```
bluesky-gym/                TU Delft BlueSky-Gym (MIT), unmodified, plus:
  solution/
    envs.py                 the competition envs with our observation, action and reward
    observations.py         intruder / obstacle / sector / route features
    probe.py                manoeuvre probe and MA shared intent
    actions.py, route.py    route-anchored action and the route planner
    shaping.py              training reward
    encoder.py              attention set encoder
    replay.py, traffic.py   prioritised replay and conflict-timed training traffic
    training.py             shared PPO setup; train_sa.py, train_ma.py
    evaluate.py             official protocol through the organisers' harness
    validate.py             parallel ranking on a validation stream
    record.py, pick_video_seeds.py   videos
    selftest.py             checks, run before anything else
submission/                 the two submitted models and their settings
results/                    official runs, validation, training logs, videos, analysis
cluster/                    PBS jobs and environment setup
```
