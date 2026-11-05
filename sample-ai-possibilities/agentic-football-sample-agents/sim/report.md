# Tuner report

- Algorithm: CMA-ES, budget 600 s (used 597 s), 24 matches/candidate/gen, popsize 10, workers 14
- Generations: 177, accepted: 6
- Held-out score (vs stock/aggressive/defensive, 60 matches, randomized): start 1.730 -> best 2.756
- Score = mean(3*win + draw + 0.3*goal_diff). Simulator is NOT the AWS engine: treat gains as hints, check on the real server.

## Points per match (15 seeds x 2 sides, domain randomization)

| opponent | before | after | before W/D/L GD | after W/D/L GD |
|---|---|---|---|---|
| stock | 1.13 | 1.14 | 4/22/4 +0 | 4/22/4 +1 |
| aggressive | 3.01 | 3.16 | 24/5/1 +44 | 25/5/0 +49 |
| defensive | 1.86 | 2.48 | 11/19/0 +13 | 18/12/0 +28 |
| tuned vs start (head-to-head) | | 1.66 | | 9/20/1 +9 |

## Changed params

| key | start | best |
|---|---|---|
| shoot_range | 0.45 | 0.7736 |
| shot_corridor | 0.06 | 0.0624 |
| gk_far_line | 0.25 | 0.0292 |
| shot_bonus | 0.0 | -0.0836 |
| pass_bonus_mid | 0.12 | 0.0019 |
| pass_bonus | 0.0 | 0.0577 |
| pass_open_min | 0.06 | 0.0435 |
| aerial_dist | 0.9 | 0.426 |
| forward_pass_weight | 0.35 | 0.1257 |
| risk_weight | 0.5 | 0.1609 |
| pressure_radius | 0.12 | -0.0053 |
| clear_zone | -0.45 | -0.3062 |
| dribble_step | 0.18 | 0.0966 |
| press_bodies | 1 | 2 |
| second_press_dist | 0.25 | 0.3223 |
| press_intensity | 0.85 | 1.5213 |
| line_height | 0.0 | -0.0296 |
| def_max_x | -0.05 | -0.0604 |
| gk_line_offset | 0.08 | 0.0936 |
| gk_leave_dist | 0.3 | 0.4856 |
| gk_mate_near | 0.15 | 0.1422 |
| stamina_sprint_min | 0.3 | 0.0974 |
| lose_shot_boost | 0.15 | 0.1677 |
| lead_risk_mult | 1.5 | 2.9101 |

Best params: `sim/params.best.json` (lib/params.json is unchanged).
