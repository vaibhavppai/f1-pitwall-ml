import os
import fastf1
import pandas as pd
from tqdm import tqdm

# Setup FastF1 cache
os.makedirs("cache", exist_ok=True)
fastf1.Cache.enable_cache("cache")

# Define seasons and rounds to fetch
rounds_by_year = {
    #2022: list(range(1, 23)),
    #2023: list(range(1, 23)),
    2024: list(range(1, 20)),
    #2025: list(range(1, 11)),
}

fp2_data = []

# Loop through each race and fetch FP2 session
for year in tqdm(rounds_by_year, desc="Fetching FP2 data"):
    for rnd in rounds_by_year[year]:
        try:
            session = fastf1.get_session(year, rnd, 'FP2')
            session.load()

            # Skip for empty laps
            if session.laps.empty:
                print(f"Skipping {year} Round {rnd} FP2 - no lap data")
                continue

            event = session.event
            best_laps = session.laps.groupby('Driver').apply(lambda x: x.pick_fastest()).reset_index(drop=True)
            session_best = best_laps['LapTime'].min()

            for _, lap in best_laps.iterrows():
                lap_time = lap['LapTime']
                gap = lap_time - session_best if pd.notnull(lap_time) else None
                fp2_data.append({
                    'Year': year,
                    'Round': rnd,
                    'GrandPrix': event['EventName'],
                    'Driver': lap['Driver'],
                    'Team': lap['Team'],
                    'FP2_LapTime': lap_time.total_seconds() if pd.notnull(lap_time) else None,
                    'FP2_GapToBest': gap.total_seconds() if pd.notnull(gap) else None
                })

        except Exception as e:
            print(f"Error in {year} Round {rnd} FP2: {e}")

# Convert to DataFrame
fp2_df = pd.DataFrame(fp2_data)

# Save to CSV
fp2_df.to_csv("fp2_best_laps_and_gaps.csv", index=False)