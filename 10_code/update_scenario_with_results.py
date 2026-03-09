import fastf1
import pandas as pd

fastf1.Cache.enable_cache("/tmp/fastf1_cache")

# Load sessions
quali = fastf1.get_session(2026, "Australia", "Q")
race = fastf1.get_session(2026, "Australia", "R")
quali.load(telemetry=False, weather=False, messages=False)
race.load(telemetry=False, weather=False, messages=False)

# Extract positions keyed by abbreviation
quali_pos = (
    quali.results[["Abbreviation", "Position"]]
    .rename(columns={"Abbreviation": "abbreviation", "Position": "qualifying_position"})
    .assign(qualifying_position=lambda df: df["qualifying_position"].astype("Int64"))
)

race_pos = (
    race.results[["Abbreviation", "Position"]]
    .rename(columns={"Abbreviation": "abbreviation", "Position": "finishing_position"})
    .assign(finishing_position=lambda df: df["finishing_position"].astype("Int64"))
)

# Load scenario file
scenario = pd.read_csv(
    "/Users/nce8/github/gridrival/00_source_data/scenario_australia_true.csv"
)

# Merge positions into driver rows
drivers = scenario[scenario["type"] == "driver"].copy()
teams = scenario[scenario["type"] == "team"].copy()

drivers = drivers.merge(quali_pos, on="abbreviation", how="left", suffixes=("_old", ""))
drivers = drivers.merge(race_pos, on="abbreviation", how="left", suffixes=("_old", ""))

# Drop old empty columns
drivers = drivers.drop(
    columns=[c for c in drivers.columns if c.endswith("_old")]
)

# Recombine and save
result = pd.concat([drivers, teams], ignore_index=True)
result.to_csv(
    "/Users/nce8/github/gridrival/00_source_data/scenario_australia_true.csv",
    index=False,
)

print(result[result["type"] == "driver"][
    ["abbreviation", "driver_name", "qualifying_position", "finishing_position"]
].to_string(index=False))


result.to_csv("../00_source_data/test_australia.csv")