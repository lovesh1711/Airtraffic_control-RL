# Results

## Official evaluation (seed 42, first 1000 episodes)

`official/final_2545470/` holds the per-episode CSVs and harness output for
the submitted models; `official/neutral_2084515/` holds the same for the
do-nothing policy (zero action: fly straight at the goal).

| Track | Policy | Intrusion [s] | Restricted [s] | Outside [s] | Sum [s] | Goal | Clean | Flight [s] |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SA | do nothing | 96.0 | 130.0 | 0.0 | 226.0 | 100% | 9.9% | 852 |
| SA | submitted | 12.4 | 1.8 | 0.1 | 14.3 | 100% | 76.8% | 1153 |
| MA | do nothing | 64.8 | 121.7 | 0.0 | 186.4 | 100% | 17.9% | 848 |
| MA | submitted | 2.9 | 2.1 | 0.0 | 5.0 | 100% | 91.5% | 1066 |

Seconds per flight (MA: per aircraft, 10 000 aircraft). Clean = goal reached
without any intrusion, restricted-area entry or sector exit. MA rows in the
CSV are per aircraft (episode = row // 10).

## Model selection (validation stream, seed 7777)

Models were chosen on seed 7777 (1000 SA / 300 MA episodes) and only the
chosen ones were run on seed 42. Rule: lowest unsafe time; models within two
standard errors of the best are a tie, broken by clean flights. All files in
`validation/` score the same episodes, so they can be merged:

    cd bluesky-gym
    python -m solution.validate --summarize ../results/validation/val_ma_*/val.json

| Candidate | Sum [s] | Clean |
| --- | ---: | ---: |
| SA `sa_2521687` (previous stage) | 15.88 | 75.7% |
| SA `sa_2527464` (+15M steps, control) | 15.29 | 75.7% |
| SA `sa_2539974` (+15M, decision every 5 s) | 16.56 | 76.0% |
| **SA `sa_2539973` (+15M, conflict-timed traffic)** | **15.02** | **77.2%** |
| MA `ma_2531101` (previous stage) | 9.00 | 87.2% |
| MA `ma_2535702` (+20M, control) | 9.46 | 86.1% |
| MA `ma_2531101` with the lateral offset, no retraining | 8.01 | 89.6% |
| **MA `ma_2539972` (+20M with the lateral offset)** | **8.04** | **89.9%** |

## Other folders

- `training/` — training log and settings (`*_meta.json`) of every stage of
  the two submitted models: SA `sa_2517964` -> `sa_2521686` -> `sa_2521687`
  -> `sa_2539973`, MA `ma_2517963` -> `ma_2521690` -> `ma_2531101` ->
  `ma_2539972`.
- `videos/video_2549833/` — the submission videos and the seed tables they
  were picked from (`seeds_*.json`).
- `analysis/` — `route_conflicts.py` (where multi-agent intrusions happen and
  what the lateral offset changes) and `traffic_counts.py` (how much harder
  the conflict-timed training traffic is). Run from `bluesky-gym/`.
