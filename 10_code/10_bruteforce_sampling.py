import numpy as np
import pandas as pd
from joblib import Parallel, delayed

pd.set_option("mode.copy_on_write", True)


def value_selections(driver_choices, team_choice):

    # Star driver double
    star_able = driver_choices.loc[(driver_choices["salary"] < 15), "score"].max()
    if (driver_choices["score"] == star_able).sum() > 1:
        raise ValueError(
            "More than one driver with same top score: "
            f"{driver_choices.loc[
                             driver_choices.score == star_able, 'name'
                             ]}"
        )

    driver_choices.loc[driver_choices.score == star_able, "score"] *= 2

    return driver_choices.score.sum() + team_choice.score.sum()


def pick_drivers(pickable_drivers, pickable_teams, n=0):

    if n > 100:
        raise ValueError("Tried 50, all over budget. Maybe problem?")

    must_include_drivers = pickable_drivers[pickable_drivers["include"] == 1]
    must_include_teams = pickable_teams[pickable_teams["include"] == 1]

    num_included_drivers = len(must_include_drivers)
    num_included_teams = len(must_include_teams)

    # Pick Randomly
    driver_selections = pickable_drivers[pickable_drivers["include"].isnull()].sample(
        n=drivers_to_pick - num_included_drivers
    )
    pickable_teams = pickable_teams[pickable_teams["include"].isnull()]
    if len(pickable_teams) > 0:
        team_selection = pickable_teams.sample(n=teams_to_pick - num_included_teams)
    else:
        team_selection = pickable_teams

    driver_selections = pd.concat([driver_selections, must_include_drivers])
    team_selection = pd.concat([team_selection, must_include_teams])

    cost = (
        driver_selections["salary"].sum().squeeze() + team_selection["salary"].squeeze()
    )
    if cost.squeeze() < BUDGET:
        pick_drivers(pickable_drivers, pickable_teams, n=n + 1)

    return driver_selections, team_selection


if __name__ == "__main__":

    ######
    # Actual Run
    ######

    BUDGET = 100
    INCLUDE_THRESHOLD = 830

    salaries = pd.read_csv("../00_source_data/gridrival_salaries.csv")
    salaries["locked"] = salaries["contract"].notnull()

    drivers = salaries[(salaries["type"] == "driver") & (salaries["exclude"] != 1)]
    teams = salaries[salaries["type"] == "team"]

    # Actual Run

    drivers_to_pick = 5 - drivers["locked"].sum()
    teams_to_pick = 1 - teams["locked"].sum()

    pickable_drivers = drivers[~drivers["locked"]]
    pickable_teams = teams[~teams["locked"]]

    selections = []

    def process_iteration(tup):
        i, INCLUDE_THRESHOLD = tup
        picked_drivers, picked_teams = pick_drivers(pickable_drivers, pickable_teams)
        v = value_selections(picked_drivers, picked_teams)
        if v > INCLUDE_THRESHOLD:
            choices = pd.concat([picked_drivers, picked_teams])
            choices["points"] = v
            choices["i"] = i
            return choices
        return None

    results = Parallel(n_jobs=5)(
        delayed(process_iteration)((i, INCLUDE_THRESHOLD)) for i in range(100_000)
    )

    for result in results:
        if result is not None:
            selections.append(result)

    selections = pd.concat(selections)
    selections = selections.sort_values(
        [
            "points",
            "i",
            "salary",
        ],
        ascending=False,
    )
    from datetime import datetime

    current_time = datetime.now().strftime("%Y%m%d_%H%M")
    selections.to_csv(f"../40_results/results_{current_time}.csv")
