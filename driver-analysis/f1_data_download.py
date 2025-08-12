import os
import fastf1
import requests
import pandas as pd

fastf1.set_log_level('WARNING')

class FastAPIDownloader:
    def __init__(self):
        pass

class RaceTrackDownloader:
    def __init__(self):
        pass

project_path = r'D:\georgiatech_ms_qcf\courses\machine_learning_cs_7641\project\f1_ml_project'
raw_data_folder = 'raw_data'
raw_data_path = os.path.join(project_path, raw_data_folder)

# --- Step 1: Caching Setup (Mandatory First Step) ---
# Define a robust cache directory path.
cache_dir = 'fastf1_cache'
cache_dir = os.path.join(project_path, cache_dir)

# Create the cache directory if it doesn't exist
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
    print(f"Created FastF1 cache directory at: {cache_dir}")

# Enable the cache.
try:
    fastf1.Cache.enable_cache(cache_dir)
    print(f"FastF1 cache enabled at: {cache_dir}")
except Exception as e:
    print(f"Error enabling FastF1 cache: {e}. Please ensure the path is valid and permissions are correct.")
    # In a production pipeline, you might raise an exception or log and exit.
    exit()

fastf1.Cache.clear_cache()

# Block to check data availability

fastf1_data_dir = 'fastf1_data'
fastf1_data_dir = os.path.join(raw_data_path, fastf1_data_dir)

# Create the cache directory if it doesn't exist
if not os.path.exists(fastf1_data_dir):
    os.makedirs(fastf1_data_dir)

# years = range(2024, 2026) # Includes 2022, 2023, 2024, 2025.
years = [2024]
session_types_to_collect = ['Q', 'S', 'SS', 'SQ', 'R']
session_types = ['Q', 'R']
all_collected_session_data = {}

print("\nInitiating data availability checks")

# Check if all event schedules are available
for year in years:

    year_folder = os.path.join(fastf1_data_dir, str(year))
    if not os.path.exists(year_folder):
        os.makedirs(year_folder)

    print(f"--- Processing Year: {year} ---")
    # all_collected_session_data[year] = {}
    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        print(f"\t Loaded event schedule for {year}. Identified {len(schedule)} Grand Prix events.")

        schedule_path = os.path.join(year_folder, f'schedule{year}.csv')
        schedule.to_csv(schedule_path)

        for index, event_row in schedule.iterrows():
            event_round = event_row['RoundNumber']
            event_name = event_row['EventName']
            event_format = event_row['EventFormat']
            print(f"\t\t -- Processing Event: Round {event_round} - {event_name} ({year}) (Format: {event_format}) --")

            # event_round = 3 # COMMENT OUT

            # TODO: MANUALLY ADJUST TO RESUME
            if event_round < 24:
                continue

            round_folder = os.path.join(year_folder, f'round{event_round}')
            if not os.path.exists(round_folder):
                os.makedirs(round_folder)

            if event_format == 'conventional':
                session_types = ['Q', 'R']
            elif event_format == 'sprint':
                session_types = ['Q', 'S', 'R']
            elif event_format == 'sprint_qualifying':
                session_types = ['Q', 'SQ', 'S', 'R']
            elif event_format == 'sprint_shootout':
                session_types = ['Q', 'SS', 'S', 'R']

            for sesh in session_types:

                sesh_folder = os.path.join(round_folder, sesh)
                if not os.path.exists(sesh_folder):
                    os.makedirs(sesh_folder)

                print(f"\t\t\t -- Processing Session: {sesh} --")
                session = fastf1.get_session(year=year, gp=event_round, identifier=sesh)
                session.load()

                sesh_drivers = session.drivers
                df_laps = session.laps

                if not df_laps.empty:
                    print("\t\t\t\t Lap data found.")
                    lap_filename = f'lap_data_{year}_round{event_round}_{sesh}.csv'
                    lap_filepath = os.path.join(sesh_folder, lap_filename)
                    df_laps.to_csv(lap_filepath)
                else:
                    print("\t\t\t\t ALERT: Lap data not found")

                actual_drivers = list(session.laps['DriverNumber'].unique())

                diff = set(sesh_drivers).difference(set(actual_drivers))
                if len(diff) != 0:
                    print(f'\t\t\t\t NOTE: All drivers in session list not there in lap time. Missing driver #s: {diff}')

                for pilot in actual_drivers:
                    # pilot = '18' # COMMENT OUT
                    telemetry_filename = f'telemetry_data_{year}_round{event_round}_{sesh}_driver{pilot}.csv'
                    telemetry_filepath = os.path.join(sesh_folder, telemetry_filename)
                    df_telemetry = pd.DataFrame()
                    fastest_lap = session.laps.pick_drivers(pilot).pick_fastest(only_by_time=False)
                    # df_telemetry = session.laps.pick_drivers(pilot).pick_quicklaps(threshold=1.001).get_telemetry()
                    if fastest_lap is None:
                        fastest_lap = session.laps.pick_drivers(pilot).pick_fastest(only_by_time=True)
                    if fastest_lap is None:
                        print(f"\t\t\t\t ALERT: Fastest lap NOT found for Driver #{pilot}")
                        # df_telemetry = pd.DataFrame()
                    else:
                        try:
                            df_telemetry = fastest_lap.get_telemetry()
                            if not df_telemetry.empty:
                                print(f"\t\t\t\t Telemetry found for Driver #{pilot}")
                            else:
                                print(f"\t\t\t\t ALERT: Telemetry NOT found for Driver #{pilot}")
                        except Exception as e:
                            print(f'\t\t\t\t TELEMETRY ERROR for Driver #{pilot}: {e}')
                    df_telemetry.to_csv(telemetry_filepath)

                for pilot in diff:
                    telemetry_filename = f'telemetry_data_{year}_round{event_round}_{sesh}_driver{pilot}.csv'
                    telemetry_filepath = os.path.join(sesh_folder, telemetry_filename)
                    df_telemetry = pd.DataFrame()
                    df_telemetry.to_csv(telemetry_filepath)

                fastf1.Cache.clear_cache()

    except Exception as e:
        # Catch any errors that prevent processing an entire year's schedule.
        print(f"**Critical Error**: An issue occurred while retrieving or processing the schedule for year {year}: {e}")
        # For detailed debugging in development, uncomment the following line:
        # import traceback; traceback.print_exc()