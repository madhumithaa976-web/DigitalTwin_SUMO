import traci
import csv
import os
from datetime import datetime, timedelta

# ---------- PATH SETTINGS ----------
SUMO_CONFIG = "../config/city.sumocfg"
OUTPUT_FILE = "../data/traffic_data.csv"

# ---------- START SUMO ----------
sumoCmd = [
    "sumo",          # use "sumo-gui" if GUI needed
    "-c", SUMO_CONFIG
]

traci.start(sumoCmd)

# ---------- CREATE DATA FOLDER ----------
os.makedirs("../data", exist_ok=True)

# ---------- INITIAL TIME SETUP ----------
start_datetime = datetime.now()

# ---------- CSV FILE SETUP ----------
with open(OUTPUT_FILE, "w", newline="") as file:
    writer = csv.writer(file)

    # Header with AM/PM time column
    writer.writerow([
        "Date",
        "Time_AM_PM",
        "Simulation_Step",
        "Vehicle_Count",
        "Average_Speed",
        "Waiting_Time"
    ])

    # ---------- SIMULATION LOOP ----------
    step = 0

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()

        vehicle_ids = traci.vehicle.getIDList()
        vehicle_count = len(vehicle_ids)

        total_speed = 0
        total_waiting = 0

        for vid in vehicle_ids:
            total_speed += traci.vehicle.getSpeed(vid)
            total_waiting += traci.vehicle.getWaitingTime(vid)

        avg_speed = total_speed / vehicle_count if vehicle_count > 0 else 0
        avg_waiting = total_waiting / vehicle_count if vehicle_count > 0 else 0

        # Calculate real timestamp
        current_time = start_datetime + timedelta(seconds=step)

        date_str = current_time.strftime("%Y-%m-%d")

        # 12-hour format with AM/PM
        time_str = current_time.strftime("%I:%M:%S %p")

        writer.writerow([
            date_str,
            time_str,
            step,
            vehicle_count,
            avg_speed,
            avg_waiting
        ])

        step += 1

# ---------- CLOSE SUMO ----------
traci.close()

print("✅ Traffic data collection completed!")
print("📁 Saved file: data/traffic_data.csv")
