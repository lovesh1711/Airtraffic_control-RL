# BlueSky-Gym ATC competition: PPO with route-anchored actions and a manoeuvre probe

This is my entry for the AI4REALNET / TU Delft BlueSky-Gym air traffic control
competition. It covers both tracks. In the single-agent track (SA) one aircraft
has to get past 10 scripted intruders; in the multi-agent track (MA) one shared
policy flies 10 aircraft at once.

## Results

Official protocol: seed 42, first 1000 episodes.

| Track | Policy | Intrusion [s] | Restricted [s] | Outside [s] | Sum [s] | Goal | Clean flights | Flight time [s] |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SA | do nothing | 96.0 | 130.0 | 0.0 | 226.0 | 100% | 9.9% | 852 |
| SA | **mine** | **12.4** | **1.8** | **0.1** | **14.3** | **100%** | **76.8%** | 1153 |
| MA | do nothing | 64.8 | 121.7 | 0.0 | 186.4 | 100% | 17.9% | 848 |
| MA | **mine** | **2.9** | **2.1** | **0.0** | **5.0** | **100%** | **91.5%** | 1066 |

Times are seconds per flight (per aircraft in MA, over 10 000 aircraft). A
clean flight reaches its goal without any intrusion, restricted-area entry or
sector exit. The per-episode CSVs are in
[`results/official/final_2545470/`](results/official/final_2545470/).

## How it works

The policy is trained with PPO from Stable-Baselines3. In the multi-agent track
every aircraft runs the same network on its own view of the world.

Other aircraft, obstacles and the candidate manoeuvres described below are
each treated as a set. Every item goes through a small shared network and
attention pools the results, so the order of the slots doesn't matter and the
same weights work for 10 intruders or 9 neighbours (`solution/encoder.py`).

The observation gives the policy what it needs to see a conflict coming:
closest point of approach to each aircraft, also along both aircraft's planned
paths, the exact obstacle polygons with time until entry, the margin to the
sector edge and time until exit, route progress, and how much time is left
before the 3000 s limit. In MA each aircraft also sees what its neighbours are
currently commanding (`solution/observations.py`).

The biggest single gain came from the manoeuvre probe. Before every decision
it takes 27 candidate commands (9 heading offsets times 3 speeds), flies each
one 240 s ahead with BlueSky's own turn and speed dynamics against where the
other traffic is going, and reports what would happen: closest approach,
seconds of intrusion, seconds inside an obstacle and seconds outside the
sector. The policy reads this as an input and still makes the decision itself
(`solution/probe.py`).

The action is a heading offset of up to 45° from the bearing to the next
waypoint of a route planned around the obstacles, plus a target speed between
200 and 317 kt. Doing nothing means following a clear path instead of flying
straight through an obstacle. When no clear route fits the time limit, the
planner accepts a longer one, and failing that the one that spends the least
distance inside obstacles (`solution/actions.py`, `solution/route.py`).

In MA, routes around the same obstacle share its corners, and that's where
most of the remaining head-on conflicts happened. So every route is flown a
few miles to the right of the shared path, the way oceanic air traffic
control does it, and opposite directions end up in separate lanes. That cut
multi-agent intrusion time by 31%.

The reward pays for progress and charges for getting close to traffic,
obstacles or the sector edge before anything is actually violated. It adds a
steep barrier just outside 5 NM, a one-off charge whenever a violation starts
and a penalty for running out of time (`solution/shaping.py`). It's only a
training signal. The scored metrics come straight from the competition
environment, which isn't modified.

During training, scenarios the policy flew badly are replayed more often. In
the last single-agent stage, half of the new scenarios have intruders re-timed
so they meet the aircraft (`solution/traffic.py`), which is allowed for
training but never used in evaluation. Every model choice was made on a
separate validation stream (seed 7777), never on the test seed 42.

## Setup

You need Python 3.12. A CPU is enough.

```bash
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
cd bluesky-gym
python -m solution.selftest      # takes a few minutes, ends with "ALL CHECKS PASSED"
```

Run everything from `bluesky-gym/` so the `bluesky_gym`, `bluesky_zoo`, `core`
and `solution` packages import from the working tree.

## Reproducing the official numbers

`solution/evaluate.py` runs the organisers' own `scripts/evaluate_competition.py`.
It only swaps in the two functions the script marks for editing (`make_env`
and `load_policy`), so seeding, the episode loop and the metrics are their
code. Each model's environment settings come from the `*_meta.json` file next
to it.

```bash
cd bluesky-gym
python -m solution.evaluate --env sa --episodes 1000 --model ../submission/sa/ppo_sa --out sa.csv
python -m solution.evaluate --env ma --episodes 1000 --model ../submission/ma/ppo_ma --out ma.csv
python -m solution.evaluate --env sa --episodes 1000 --out neutral_sa.csv     # the do-nothing baseline
```

These run one episode after another, so SA takes about an hour and MA five to
six. If you just want a quick check, `validate.py` flies the same episodes in
parallel:

```bash
python -m solution.validate --kind sa --episodes 1000 --eval-seed 42 --workers 8 --out sa42.json ../submission/sa/ppo_sa.zip
python -m solution.validate --kind ma --episodes 1000 --eval-seed 42 --workers 8 --out ma42.json ../submission/ma/ppo_ma.zip
```

Expect small differences between machines because of floating-point rounding.
Most single-agent episodes come out identical. With ten aircraft reacting to
each other the multi-agent mean can move by about a second.

## Videos

I didn't pick the five scenarios by eye. Out of scenario seeds 1 to 40, the
script keeps those my policy flies cleanly and takes the five where doing
nothing goes worst.

```bash
python -m solution.pick_video_seeds --env sa --model ../submission/sa/ppo_sa --out seeds_sa.json
python -m solution.record --env sa --model ../submission/sa/ppo_sa --seeds 9 8 13 2 33 --out reel_sa.gif
python -m solution.record --env sa --model ../submission/sa/ppo_sa --seeds 9 8 13 2 33 \
    --render-size 640 --caption --compare-neutral --size 900 --max-frames 500 --out reel_sa_vs_neutral.gif
```

The submitted videos are in [`results/videos/video_2549833/`](results/videos/video_2549833/)
(SA seeds 9 8 13 2 33, MA seeds 2 33 19 13 40).

## Training

I trained on a PBS cluster in stages, each one continuing from the previous
model, using the job scripts in [`cluster/`](cluster/). `cluster/setup.sh`
builds the environment there. Submit from the repository root. `<id>` stands
for the job id of the previous stage, whose output goes to
`results/<track>_<id>/`. The times are for 20 cores.

```bash
Q="-l select=1:ncpus=20"

# single-agent: 15M steps fresh, two 15M stages with the fallback routing
# (learning rate 3e-4 to 1.5e-4 to 0), then 15M with re-timed traffic. About 12 h each.
qsub $Q -v TRACK=sa,STAGE=1,TIMESTEPS=15000000,SEED=170 cluster/train.pbs
qsub $Q -v TRACK=sa,STAGE=2,TIMESTEPS=15000000,LR_FINAL=0.00015,SEED=180,RESUME=results/sa_<id>/ppo_sa.zip cluster/train.pbs
qsub $Q -v TRACK=sa,STAGE=2,TIMESTEPS=15000000,LR=0.00015,LR_FINAL=0,SEED=181,RESUME=results/sa_<id>/ppo_sa.zip cluster/train.pbs
qsub $Q -v TRACK=sa,STAGE=2,TIMESTEPS=15000000,LR=0.0001,LR_FINAL=0,HARD_TRAFFIC_PROB=0.5,SEED=200,RESUME=results/sa_<id>/ppo_sa.zip cluster/train.pbs

# multi-agent: 20M fresh (about 14 h), then three 20M stages (about 8 h each),
# the last one with the lateral offset
qsub $Q -l walltime=23:30:00 -v TRACK=ma,STAGE=1,TIMESTEPS=20000000,SEED=121 cluster/train.pbs
qsub $Q -v TRACK=ma,STAGE=2,TIMESTEPS=20000000,LR_FINAL=0,SEED=184,RESUME=results/ma_<id>/ppo_ma.zip cluster/train.pbs
qsub $Q -v TRACK=ma,STAGE=2,TIMESTEPS=20000000,LR_FINAL=0,SEED=190,RESUME=results/ma_<id>/ppo_ma.zip cluster/train.pbs
qsub $Q -v TRACK=ma,STAGE=2,TIMESTEPS=20000000,LR_FINAL=0,ROUTE_LATERAL_OFFSET_NM=3,SEED=202,RESUME=results/ma_<id>/ppo_ma.zip cluster/train.pbs
```

`cluster/train.pbs` shows the full set of flags. Without PBS, the same runs are
`python -m solution.train_sa ...` and `python -m solution.train_ma ...` (try
`--help`). The trainers refuse any seed that would put a rollout worker on
seed 42, the test stream. To compare candidates, use `cluster/validate.pbs`,
then run the winners with `cluster/evaluate.pbs`. The training logs of every
stage are in [`results/training/`](results/training/).

## What's where

```
bluesky-gym/                TU Delft's BlueSky-Gym (MIT licence), unmodified, plus:
  solution/
    envs.py                 the competition envs with my observation, action and reward
    observations.py         intruder, obstacle, sector and route features
    probe.py                manoeuvre probe and MA shared intent
    actions.py, route.py    route-anchored action and the route planner
    shaping.py              training reward
    encoder.py              attention set encoder
    replay.py, traffic.py   scenario replay and re-timed training traffic
    training.py             shared PPO setup used by train_sa.py and train_ma.py
    evaluate.py             official protocol through the organisers' harness
    validate.py             parallel comparison on a validation stream
    record.py, pick_video_seeds.py   videos
    selftest.py             quick checks, worth running first
submission/                 the two submitted models and their settings
results/                    official runs, validation, training logs, videos, analysis
cluster/                    PBS job scripts and environment setup
```
