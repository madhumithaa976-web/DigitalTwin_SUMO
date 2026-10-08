import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

rows = 2500

vehicle_count = []
average_speed = []
waiting_time = []
dates = []
times = []

# Start date & time
start_datetime = datetime(2026, 1, 1, 6, 0, 0)  # Jan 1, 2026 - 6:00 AM

for i in range(rows):

    current_time = start_datetime + timedelta(minutes=i)

    # Store Date
    dates.append(current_time.strftime("%d-%m-%Y"))

    # Store Time with AM/PM
    times.append(current_time.strftime("%I:%M:%S %p"))

    # Simulate peak hours (Morning & Evening Rush)
    if 150 < i < 250 or 350 < i < 450:
        traffic = np.random.randint(80, 120)
    else:
        traffic = np.random.randint(40, 80)

    # Speed decreases when traffic increases
    speed = 80 - (traffic * 0.5) + np.random.randint(-5, 5)
    speed = max(15, min(speed, 80))

    # Waiting time increases with traffic
    waiting = traffic * 0.3 + np.random.randint(0, 5)

    vehicle_count.append(traffic)
    average_speed.append(round(speed, 2))
    waiting_time.append(round(waiting, 2))

data = pd.DataFrame({
    "Date": dates,
    "Time": times,
    "Vehicle_Count": vehicle_count,
    "Average_Speed": average_speed,
    "Waiting_Time": waiting_time
})

data.to_csv("../module3_preprocessing/processed_data.csv", index=False)

print("✅ Correct realistic traffic dataset with Date & AM/PM Time created!")
print("📁 Saved to: module3_preprocessing/processed_data.csv")