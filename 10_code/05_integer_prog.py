import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import optimize
from scipy.optimize import milp

pd.set_option("mode.copy_on_write", True)

BUDGET = 100

salaries = pd.read_csv("../00_source_data/gridrival_stats.csv")

# Drop people I hate
salaries["locked"] = salaries["contract"].notnull()
salaries = salaries[(salaries["exclude"] != 1)]
locked = salaries[salaries["locked"]]

# Figure out who is in contract.
drivers_to_pick = 5 - salaries.loc[salaries.type == "driver", "locked"].sum()
teams_to_pick = 1 - salaries.loc[salaries.type == "team", "locked"].sum()

# Pull Out picks
pre_picked = salaries[salaries["include"] == 1]
actual_budget = BUDGET - pre_picked.salary.sum()

drivers_to_pick -= pre_picked.loc[
    (pre_picked["include"] == 1) & (pre_picked["type"] == "driver"), "include"
].sum()

teams_to_pick -= pre_picked.loc[
    (pre_picked["include"] == 1) & (pre_picked["type"] == "team"), "include"
].sum()

# Get actually selectable.
pickable = salaries[(~salaries["locked"]) & (salaries["include"] != 1)]

if teams_to_pick == 0:
    pickable = pickable[pickable["type"] != "team"]

if drivers_to_pick == 0:
    pickable = pickable[pickable["type"] != "driver"]

pickable = pickable.sort_values("type").reset_index(drop=True)


#############
# optimize
#############

salaries = pickable.salary.values
points = pickable.score.values

bounds = optimize.Bounds(0, 1)
integrality = np.full_like(points, True)

##
# Constraints
##

constraints = []

# Budget constraint
constraints.append(optimize.LinearConstraint(A=salaries, lb=0, ub=actual_budget))

# One Team
if teams_to_pick == 1:
    teams_start = pickable[pickable["type"] == "team"].index[0]
    team_A = np.zeros(len(pickable))
    team_A[teams_start:] = 1
    team_A
    constraints.append(optimize.LinearConstraint(A=team_A, lb=0, ub=teams_to_pick))

# Five drivers
if drivers_to_pick > 0:
    drivers_A = np.zeros(len(pickable))
    if teams_to_pick == 1:
        drivers_A[:teams_start] = 1
    else:
        drivers_A[:] = 1
    drivers_A
    constraints.append(optimize.LinearConstraint(A=drivers_A, lb=0, ub=drivers_to_pick))

###
# Run Optimization for Best Picks
###

# Star driver is hard to do in integer programming framework, so
# just loop over each possible star, optimize, and pick best at the
# end.
#
# Adding a -1 index for the version without a starred driver in case
# best starred driver is one under contract or pre-picked.

starable_drivers = pickable[
    (pickable["type"] == "driver") & (pickable["salary"] <= 15)
].index.append(pd.Index([-1]))

optimal_picks = []

for i in starable_drivers:
    points_copy = points.copy()

    if i != -1:
        points_copy[i] *= 2

    picks = milp(
        c=-points_copy,
        constraints=constraints,
        integrality=integrality,
        bounds=bounds,
    )
    picked_choices = pickable[picks.x.astype("bool")]
    picked_choices["starred"] = 0
    picked_choices["overall_value"] = -picks.fun
    picked_choices.loc[picked_choices.index == i, "starred"] = 1
    picked_choices["starred_index"] = i
    optimal_picks.append(picked_choices)

picks = pd.concat(optimal_picks)
picks = picks.sort_values(["overall_value", "starred_index"], ascending=False)
best_w_starred = picks[picks["starred_index"] == picks.iloc[0]["starred_index"]]
best_wo_starred = picks[picks["starred_index"] == -1]

#######
# Pull back in people under contract or
# hand-picked.
#
# Check whether better to have starred driver in set from
# optimizer or not and keep only best.
#######

# Add back in pre-picked and contracts if starred in optimized set.
full_w_starred = pd.concat([best_w_starred, pre_picked, locked])
full_w_starred["starred"] = full_w_starred["starred"].fillna(0)

# Add pre-picked and contracts if star not in selections from optimizer
full_wo_starred = pd.concat([best_wo_starred, pre_picked, locked])
full_wo_starred["starred"] = full_wo_starred["starred"].fillna(0)

starrable_in_full = full_wo_starred[(full_wo_starred["salary"] <= 15)]
assert (
    len(
        starrable_in_full[
            starrable_in_full["score"] == starrable_in_full["score"].max()
        ]
    )
    <= 1
)

full_wo_starred.loc[
    (full_wo_starred["score"] == starrable_in_full["score"].max())
    & (full_wo_starred["salary"] <= 15),
    "starred",
] = 1

full_wo_starred.loc[
    (full_wo_starred["score"] == starrable_in_full["score"].max())
    & (full_wo_starred["salary"] <= 15),
    "score",
] *= 2

# Better to star an optimizer picked driver or
# someone pre-picked or under contract.
if full_wo_starred.score.sum() >= full_w_starred.score.sum():
    best = full_wo_starred
else:
    best = full_w_starred

#######
# Gather and save
#######

best = best.sort_values(
    ["type", "score", "salary", "starred"],
    ascending=False,
)

print(
    f"best: {best["score"].sum()}. \n "
    f"{best[["type", "name", "salary", "score", "starred"]]}"
)

from datetime import datetime

current_time = datetime.now().strftime("%Y_%m_%d_%H_%M")
best[["type", "name", "salary", "score", "starred", "include"]].to_csv(
    f"../40_results/results_{current_time}.csv"
)
