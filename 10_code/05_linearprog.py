import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import optimize

pd.set_option("mode.copy_on_write", True)

BUDGET = 100

salaries = pd.read_csv("../00_source_data/gridrival_stats.csv")

# Drop people I hate
salaries["locked"] = salaries["contract"].notnull()
salaries = salaries[(salaries["exclude"] != 1)]

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
    drivers_A[:teams_start] = 1
    drivers_A
    constraints.append(optimize.LinearConstraint(A=drivers_A, lb=0, ub=drivers_to_pick))

from scipy.optimize import milp

picks = milp(
    c=-points,
    constraints=constraints,
    integrality=integrality,
    bounds=bounds,
)
picks = pickable[picks.x.astype("bool")]

#######
# Gather and save
#######

selections = pd.concat([picks, pre_picked])
selections = selections.sort_values(
    [
        "type",
        "score",
        "salary",
    ],
    ascending=False,
)
from datetime import datetime

current_time = datetime.now().strftime("%Y_%m_%d_%H_%M")
selections[["type", "name", "salary", "score", "include"]].to_csv(
    f"../40_results/results_{current_time}.csv"
)

selections
