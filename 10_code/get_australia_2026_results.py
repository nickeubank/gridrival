import fastf1
import pandas as pd

fastf1.Cache.enable_cache("/tmp/fastf1_cache")

# Load qualifying and race sessions
quali = fastf1.get_session(2026, "Australia", "Q")
race = fastf1.get_session(2026, "Australia", "R")

quali.load(telemetry=False, weather=False, messages=False)
race.load(telemetry=False, weather=False, messages=False)

# --- Qualifying results ---
quali_results = quali.results[
    [
        "DriverNumber",
        "Abbreviation",
        "FullName",
        "TeamName",
        "Q1",
        "Q2",
        "Q3",
        "Position",
    ]
].copy()
quali_results = quali_results.sort_values("Position").reset_index(drop=True)

print("=== 2026 Australian GP — Qualifying ===")
print(quali_results.to_string(index=False))

# --- Race results ---
race_results = race.results[
    ["DriverNumber", "Abbreviation", "FullName", "TeamName", "Position"]
].copy()
race_results = race_results.sort_values("Position").reset_index(drop=True)

print("\n=== 2026 Australian GP — Race ===")
print(race_results.to_string(index=False))
