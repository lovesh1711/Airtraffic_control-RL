# Results

## Official evaluation (seed 42, first 1000 episodes)

`official/final_2545470/` has the per-episode CSVs and harness output for the
submitted models. `official/neutral_2084515/` has the same for the do-nothing
policy, which just flies straight at the goal.

| Track | Policy | Intrusion [s] | Restricted [s] | Outside [s] | Sum [s] | Goal | Clean | Flight [s] |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SA | do nothing | 96.0 | 130.0 | 0.0 | 226.0 | 100% | 9.9% | 852 |
| SA | submitted | 12.4 | 1.8 | 0.1 | 14.3 | 100% | 76.8% | 1153 |
| MA | do nothing | 64.8 | 121.7 | 0.0 | 186.4 | 100% | 17.9% | 848 |
| MA | submitted | 2.9 | 2.1 | 0.0 | 5.0 | 100% | 91.5% | 1066 |

Times are seconds per flight (per aircraft in MA, 10 000 aircraft). A clean
flight reaches its goal without any intrusion, restricted-area entry or sector
exit. In the MA CSV each row is one aircraft, so the episode is row // 10.

## How the models were picked

I compared candidates on a different seed (7777, 1000 SA or 300 MA episodes)
and only ran the chosen ones on seed 42. The rule was fixed beforehand: lowest
unsafe time wins, anything within two standard errors of the best counts as a
tie, and ties go to the higher share of clean flights. All files in
`validation/` use the same episodes, so you can merge them:

    cd bluesky-gym
    python -m solution.validate --summarize ../results/validation/val_ma_*/val.json

| Candidate | Sum [s] | Clean |
| --- | ---: | ---: |
| SA `sa_2521687` (previous stage) | 15.88 | 75.7% |
| SA `sa_2527464` (15M more steps, as a control) | 15.29 | 75.7% |
| SA `sa_2539974` (15M more, deciding every 5 s) | 16.56 | 76.0% |
| SA `sa_2539973` (15M more, re-timed traffic), submitted | 15.02 | 77.2% |
| MA `ma_2531101` (previous stage) | 9.00 | 87.2% |
| MA `ma_2535702` (20M more, as a control) | 9.46 | 86.1% |
| MA `ma_2531101` with the lateral offset, no retraining | 8.01 | 89.6% |
| MA `ma_2539972` (20M more with the lateral offset), submitted | 8.04 | 89.9% |

## The other folders

`training/` has the training log and settings (`*_meta.json`) for every stage
of both submitted models. The SA line is `sa_2517964`, `sa_2521686`,
`sa_2521687`, `sa_2539973`; the MA line is `ma_2517963`, `ma_2521690`,
`ma_2531101`, `ma_2539972`.

`videos/video_2549833/` has the submission videos and the seed tables they
were picked from (`seeds_*.json`).

`analysis/` has two scripts, run from `bluesky-gym/`. `route_conflicts.py`
shows where the multi-agent intrusions happen and what the lateral offset
changes. `traffic_counts.py` measures how much harder the re-timed training
traffic is.
