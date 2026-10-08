import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import warnings
import time
import os
warnings.filterwarnings('ignore')

# ============================================================
# SCRIPT START - USE CURRENT DATE & TIME
# ============================================================
script_start_time = datetime.now()
script_start_timestamp = time.time()

print("="*60)
print("TRAFFIC PREDICTION SYSTEM - CURRENT TIME BASED")
print("="*60)

# Format current date and time
current_date = script_start_time.strftime('%A, %B %d, %Y')
current_time_12h = script_start_time.strftime('%I:%M:%S %p')
current_time_24h = script_start_time.strftime('%H:%M:%S')
current_hour = script_start_time.strftime('%I:%M %p')
current_hour_24h = script_start_time.strftime('%H:%M')

print(f"\nCURRENT DATE: {current_date}")
print(f"CURRENT TIME: {current_time_12h} ({current_time_24h} 24h)")
print(f"LOCATION: Pollachi City Center")
print(f"\nUsing CURRENT TIME for prediction...")

# ============================================================
# 1. LOAD MODEL AND DATA
# ============================================================
print("\n" + "-"*60)
print("STEP 1: LOADING MODEL AND DATA")
print("-"*60)

load_start = time.time()

try:
    # Load model and scaler
    model = load_model("module4_lstm_prediction/lstm_model.keras")
    scaler = joblib.load("scaler.pkl")
    print("✓ Model and scaler loaded successfully")
    
    # Load data
    csv_path = "C:/Users/MADHUMITHA A/OneDrive/Desktop/DigitalTwin_SUMO/module3_preprocessing/processed_data.csv"
    df = pd.read_csv(csv_path)
    
    features = ["Average_Speed", "Vehicle_Count", "Waiting_Time"]
    original_data = df[features].values
    
    load_end = time.time()
    load_duration = load_end - load_start
    
    print(f"✓ Data loaded: {len(original_data)} rows")
    print(f"✓ Features: Speed | Vehicle Count | Waiting Time")
    print(f"✓ Loading time: {load_duration:.2f} seconds")
    
except Exception as e:
    print(f"✗ Error loading data: {e}")
    exit()

# ============================================================
# 2. FIND DATA MATCHING CURRENT TIME
# ============================================================
print("\n" + "-"*60)
print("STEP 2: FINDING DATA FOR CURRENT TIME")
print("-"*60)

# Create datetime series for all data
base_datetime = script_start_time.replace(minute=0, second=0, microsecond=0)
df["Datetime"] = [base_datetime + timedelta(minutes=i) for i in range(len(df))]

# Extract current hour and minute
current_hour_num = script_start_time.hour
current_minute = script_start_time.minute

print(f"\nCurrent Time Analysis:")
print(f"   • Hour: {current_hour_num:02d}:00")
print(f"   • Minute: {current_minute:02d}")
print(f"   • Looking for data at: {script_start_time.strftime('%I:%M %p')}")

# Find the index closest to current time
target_minute = current_minute
start_index = target_minute

# Ensure we have enough data (need 10 past + 5 future = 15 minutes)
if start_index < 10:
    print(f"⚠ Not enough past data at current minute. Using earliest available.")
    start_index = 10
elif start_index > len(original_data) - 15:
    print(f"⚠ Not enough future data. Using latest available.")
    start_index = len(original_data) - 15

# Get the exact time window
start = start_index - 10  # Go back 10 minutes to get past data

# Extract data for the window
past_10 = original_data[start:start+10]        # 10 minutes PAST data
actual_future = original_data[start+10:start+15]  # 5 minutes FUTURE data

# Get corresponding times
window_times = df["Datetime"].iloc[start:start+15]
window_times_12h = window_times.dt.strftime('%I:%M %p').values
window_times_24h = window_times.dt.strftime('%H:%M').values
window_dates = window_times.dt.strftime('%A, %B %d, %Y').values

# Find current time index (minute 10 in our window)
current_time_index = 9  # Index 9 is the 10th minute (current time)
current_time_str = window_times_12h[current_time_index]

print(f"\nSelected Time Window based on CURRENT TIME:")
print(f"   Date: {window_dates[0]}")
print(f"   Window: {window_times_12h[0]} to {window_times_12h[14]}")
print(f"   CURRENT TIME: {current_time_str} (Minute 10 in window)")
print(f"\nData Split:")
print(f"   • PAST 10 minutes: {window_times_12h[0]} to {window_times_12h[9]} (INPUT)")
print(f"   • FUTURE 5 minutes: {window_times_12h[10]} to {window_times_12h[14]} (OUTPUT)")

# Display the past 10 minutes data
print("\nPAST 10 MINUTES DATA (Input to Model):")
print(f"{'Minute':<8} {'Time (12h)':<12} {'Time (24h)':<10} {'Speed':<8} {'VEHICLES':<10} {'Wait':<8} {'Status':<10}")
print("-"*70)
for i in range(10):
    status = "CURRENT" if i == 9 else f"{10-i} min ago"
    print(f"{i+1:<8} {window_times_12h[i]:<12} {window_times_24h[i]:<10} {past_10[i][0]:<8.1f} {past_10[i][1]:<10.0f} {past_10[i][2]:<8.1f} {status:<10}")

# ============================================================
# 3. ANALYZE DATA DISTRIBUTION
# ============================================================
print("\n" + "-"*60)
print("STEP 3: ANALYZING TRAFFIC PATTERNS")
print("-"*60)

analyze_start = time.time()

# Calculate percentiles for thresholds
speed_percentiles = np.percentile(original_data[:,0], [20, 40, 60, 80])
wait_percentiles = np.percentile(original_data[:,2], [20, 40, 60, 80])
count_percentiles = np.percentile(original_data[:,1], [20, 40, 60, 80])

analyze_end = time.time()
analyze_duration = analyze_end - analyze_start

print(f"\nVehicle Count Statistics:")
print(f"   • Minimum: {original_data[:,1].min():.0f} vehicles")
print(f"   • Maximum: {original_data[:,1].max():.0f} vehicles")
print(f"   • Average: {original_data[:,1].mean():.1f} vehicles")
print(f"\nTraffic Thresholds:")
print(f"   • Low: < {count_percentiles[1]:.0f} vehicles")
print(f"   • Medium: {count_percentiles[1]:.0f} - {count_percentiles[2]:.0f}")
print(f"   • High: {count_percentiles[2]:.0f} - {count_percentiles[3]:.0f}")
print(f"   • Peak: > {count_percentiles[3]:.0f} vehicles")
print(f"\nAnalysis time: {analyze_duration:.2f} seconds")

# ============================================================
# 4. CONGESTION FUNCTION
# ============================================================
def get_congestion_level(speed, wait, vehicle_count):
    """Enhanced congestion detection using all three parameters"""
    
    # Base congestion from speed and wait
    if speed <= speed_percentiles[0] and wait >= wait_percentiles[2]:
        base_level = "HIGH"
    elif speed <= speed_percentiles[1] or wait >= wait_percentiles[2]:
        base_level = "MEDIUM-HIGH"
    elif speed <= speed_percentiles[2] and wait >= wait_percentiles[1]:
        base_level = "MEDIUM-LOW"
    elif speed <= speed_percentiles[3] and wait >= wait_percentiles[0]:
        base_level = "MEDIUM"
    else:
        base_level = "LOW"
    
    # Enhance with vehicle count
    if vehicle_count >= count_percentiles[3]:
        count_indicator = " [PEAK]"
    elif vehicle_count >= count_percentiles[2]:
        count_indicator = " [HIGH]"
    elif vehicle_count >= count_percentiles[1]:
        count_indicator = " [MEDIUM]"
    else:
        count_indicator = " [LOW]"
    
    return f"{base_level}{count_indicator}"

# ============================================================
# 5. MAKE PREDICTIONS
# ============================================================
print("\n" + "-"*60)
print("STEP 4: PREDICTING NEXT 5 MINUTES")
print(f"   Using PAST 10 minutes ({window_times_12h[0]} to {window_times_12h[9]})")
print(f"   Predicting FUTURE 5 minutes ({window_times_12h[10]} to {window_times_12h[14]})")
print("-"*60)

pred_start = time.time()
print(f"Prediction started at: {datetime.now().strftime('%I:%M:%S %p')}")

# Scale and predict
scaled_past = scaler.transform(past_10).reshape(1, 10, 3)
prediction = model.predict(scaled_past, verbose=0)

# Inverse transform predictions
future_pred = scaler.inverse_transform(prediction.reshape(5, 3))
future_pred = np.maximum(0, future_pred)

pred_end = time.time()
pred_duration = pred_end - pred_start

print(f"✓ Predictions completed successfully")
print(f"Prediction time: {pred_duration:.2f} seconds")

# ============================================================
# 6. CALCULATE CONGESTION LEVELS AND TRENDS
# ============================================================
congestion_levels = []
vehicle_trends = []

for i in range(5):
    level = get_congestion_level(future_pred[i][0], future_pred[i][2], future_pred[i][1])
    congestion_levels.append(level)
    
    # Determine vehicle count trend
    if i > 0:
        if future_pred[i][1] > future_pred[i-1][1]:
            vehicle_trends.append("INCREASING")
        elif future_pred[i][1] < future_pred[i-1][1]:
            vehicle_trends.append("DECREASING")
        else:
            vehicle_trends.append("STABLE")
    else:
        vehicle_trends.append("STABLE")

# ============================================================
# 7. DISPLAY RESULTS
# ============================================================
print("\n" + "="*80)
print("STEP 5: CURRENT TIME PREDICTION RESULTS")
print("="*80)
print(f"\nLOCATION: Pollachi City Center")
print(f"DATE: {window_dates[9]}")
print(f"CURRENT TIME: {window_times_12h[9]} ({window_times_24h[9]} 24h)")
print(f"SCRIPT RUN TIME: {datetime.now().strftime('%I:%M:%S %p')}")
print("="*80)

# Past 10 minutes display
print("\nPAST 10 MINUTES - HISTORICAL DATA:")
print(f"{'No.':<4} {'Time (12h)':<12} {'Time (24h)':<10} {'Speed':<8} {'VEHICLES':<10} {'Wait':<8} {'Status':<12} {'Traffic Level':<25}")
print("-"*95)

for i in range(10):
    level = get_congestion_level(past_10[i][0], past_10[i][2], past_10[i][1])
    status = "CURRENT NOW" if i == 9 else f"{10-i} min ago"
    print(f"{i+1:<4} {window_times_12h[i]:<12} {window_times_24h[i]:<10} {past_10[i][0]:<8.1f} {past_10[i][1]:<10.0f} {past_10[i][2]:<8.1f} {status:<12} {level:<25}")

# Future 5 minutes predicted display
print("\nFUTURE 5 MINUTES - PREDICTED TRAFFIC:")
print(f"{'No.':<4} {'Time (12h)':<12} {'Time (24h)':<10} {'Speed':<8} {'VEHICLES':<10} {'Wait':<8} {'From Now':<10} {'Traffic Level':<25}")
print("-"*95)

for i in range(5):
    level = congestion_levels[i]
    minutes_from_now = f"T+{i+1}"
    print(f"{i+11:<4} {window_times_12h[i+10]:<12} {window_times_24h[i+10]:<10} {future_pred[i][0]:<8.1f} {future_pred[i][1]:<10.0f} {future_pred[i][2]:<8.1f} {minutes_from_now:<10} {level:<25}")

# ============================================================
# 8. VEHICLE COUNT ANALYSIS
# ============================================================
print("\n" + "="*60)
print("STEP 6: CURRENT TIME VEHICLE COUNT ANALYSIS")
print("="*60)

analysis_start = time.time()

# Calculate statistics
current_count = past_10[-1][1]  # Current vehicle count
avg_past_count = np.mean(past_10[:,1])
avg_future_count = np.mean(future_pred[:,1])
max_future_count = np.max(future_pred[:,1])
min_future_count = np.min(future_pred[:,1])
peak_idx = np.argmax(future_pred[:,1])

analysis_end = time.time()
analysis_duration = analysis_end - analysis_start

print(f"\nCURRENT STATUS (at {window_times_12h[9]}):")
print(f"   • Vehicle Count: {current_count:.0f} vehicles")
print(f"   • Speed: {past_10[-1][0]:.1f} km/h")
print(f"   • Wait Time: {past_10[-1][2]:.1f} seconds")
print(f"   • Condition: {get_congestion_level(past_10[-1][0], past_10[-1][2], past_10[-1][1])}")

print(f"\nPAST 10 MINUTES SUMMARY:")
print(f"   • Average Count: {avg_past_count:.1f} vehicles")
print(f"   • Minimum: {np.min(past_10[:,1]):.0f} vehicles")
print(f"   • Maximum: {np.max(past_10[:,1]):.0f} vehicles")

print(f"\nFUTURE 5 MINUTES PREDICTION:")
print(f"   • Average Count: {avg_future_count:.1f} vehicles")
print(f"   • Range: {min_future_count:.0f} - {max_future_count:.0f} vehicles")
print(f"   • Change: {((avg_future_count - avg_past_count)/avg_past_count*100):+.1f}%")

# Peak prediction
peak_time = window_times_12h[peak_idx + 10]
print(f"   • Expected Peak: {max_future_count:.0f} vehicles at {peak_time}")
print(f"   • Peak is {peak_idx + 1} minutes from now")

# Warning if peak exceeds threshold
if max_future_count > count_percentiles[3]:
    print(f"\n*** PEAK TRAFFIC WARNING ***")
    print(f"   • Peak will exceed {count_percentiles[3]:.0f} vehicles")
    print(f"   • Be prepared for heavy traffic at {peak_time}")

print(f"\nAnalysis time: {analysis_duration:.2f} seconds")

# ============================================================
# 9. OVERALL TRAFFIC SUMMARY
# ============================================================
print("\n" + "="*60)
print("STEP 7: OVERALL TRAFFIC SUMMARY")
print("="*60)

avg_speed = np.mean(future_pred[:,0])
avg_wait = np.mean(future_pred[:,2])
overall = get_congestion_level(avg_speed, avg_wait, avg_future_count)

print(f"\nLocation: Pollachi City Center")
print(f"Date: {window_dates[9]}")
print(f"Current Time: {window_times_12h[9]}")
print(f"\nNext 5 Minutes Forecast:")
print(f"   • Expected Average Speed: {avg_speed:.1f} km/h")
print(f"   • Expected Average Count: {avg_future_count:.1f} vehicles")
print(f"   • Expected Average Wait: {avg_wait:.1f} seconds")
print(f"\nOverall Condition: {overall}")
print(f"\nTime Progression:")
time_progression = ""
for i in range(5):
    time_progression += f"{window_times_12h[i+10]}"
    if i < 4:
        time_progression += " → "
print(f"   {time_progression}")

# ============================================================
# 10. GENERATE PLOTS (FIXED VERSION)
# ============================================================
print("\n" + "-"*60)
print("STEP 8: GENERATING VISUALIZATIONS")
print("-"*60)

plot_start = time.time()

# Set larger figure size and better layout
plt.figure(figsize=(16, 10))
plt.suptitle(
    f'Traffic Prediction - Current Time: {window_times_12h[9]}\n'
    f'Location: Pollachi City Center | Date: {window_dates[9]}',
    fontsize=16,
    fontweight='bold',
    y=0.98
)

# [1] Speed Plot
ax1 = plt.subplot(2,3,1)
ax1.plot(range(1,11), past_10[:,0], 'b-o', label='Past Speed', linewidth=2, markersize=6)
ax1.plot(range(11,16), future_pred[:,0], 'r--s', label='Predicted', linewidth=2, markersize=8)
ax1.axvline(x=9.5, color='green', linestyle='-', linewidth=3, alpha=0.5, label='CURRENT TIME')
ax1.set_title("Speed Prediction", fontweight='bold', fontsize=12)
ax1.set_xlabel("Minutes", fontsize=10)
ax1.set_ylabel("Speed (km/h)", fontsize=10)
ax1.legend(loc='best', fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(range(1,16))
ax1.set_xticklabels([f"{i}\n{window_times_12h[i-1][:5]}" for i in range(1,16)], rotation=45, fontsize=8)

# [2] Vehicle Count Plot
ax2 = plt.subplot(2,3,2)
ax2.plot(range(1,11), past_10[:,1], 'b-o', label='Past Count', linewidth=2, markersize=6)
ax2.plot(range(11,16), future_pred[:,1], 'r--s', label='Predicted', linewidth=2, markersize=8)
ax2.axvline(x=9.5, color='green', linestyle='-', linewidth=3, alpha=0.5, label='CURRENT TIME')
ax2.axhline(y=count_percentiles[3], color='red', linestyle=':', alpha=0.5, label='Peak Threshold')
ax2.set_title("Vehicle Count Prediction", fontweight='bold', fontsize=12)
ax2.set_xlabel("Minutes", fontsize=10)
ax2.set_ylabel("Number of Vehicles", fontsize=10)
ax2.legend(loc='best', fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(range(1,16))
ax2.set_xticklabels([f"{i}\n{window_times_12h[i-1][:5]}" for i in range(1,16)], rotation=45, fontsize=8)

# [3] Waiting Time Plot
ax3 = plt.subplot(2,3,3)
ax3.plot(range(1,11), past_10[:,2], 'b-o', label='Past Wait', linewidth=2, markersize=6)
ax3.plot(range(11,16), future_pred[:,2], 'r--s', label='Predicted', linewidth=2, markersize=8)
ax3.axvline(x=9.5, color='green', linestyle='-', linewidth=3, alpha=0.5, label='CURRENT TIME')
ax3.set_title("Waiting Time Prediction", fontweight='bold', fontsize=12)
ax3.set_xlabel("Minutes", fontsize=10)
ax3.set_ylabel("Wait Time (seconds)", fontsize=10)
ax3.legend(loc='best', fontsize=8)
ax3.grid(True, alpha=0.3)
ax3.set_xticks(range(1,16))
ax3.set_xticklabels([f"{i}\n{window_times_12h[i-1][:5]}" for i in range(1,16)], rotation=45, fontsize=8)

# [4] Vehicle Count Bar Chart
ax4 = plt.subplot(2,3,4)
colors = ['blue']*9 + ['green'] + ['red']*5
bars = ax4.bar(range(1,16), 
               np.concatenate([past_10[:,1], future_pred[:,1]]),
               color=colors, alpha=0.7)
ax4.set_title("Vehicle Count Distribution", fontweight='bold', fontsize=12)
ax4.set_xlabel("Minutes", fontsize=10)
ax4.set_ylabel("Vehicle Count", fontsize=10)
ax4.axvline(x=9.5, color='black', linestyle='--', alpha=0.5)
ax4.set_xticks(range(1,16))
ax4.set_xticklabels([f"{i}\n{window_times_12h[i-1][:5]}" for i in range(1,16)], rotation=45, fontsize=8)

# Add value labels with adjusted positions
for i, bar in enumerate(bars):
    height = bar.get_height()
    color = 'white' if i >= 10 else 'black'
    ax4.text(bar.get_x() + bar.get_width()/2., height/2,
             f'{int(height)}', ha='center', va='center', color=color, fontsize=7, fontweight='bold')

# [5] Congestion Levels
ax5 = plt.subplot(2,3,5)
future_colors = []
for level in congestion_levels:
    if 'HIGH' in level:
        future_colors.append('red')
    elif 'MEDIUM' in level:
        future_colors.append('orange')
    else:
        future_colors.append('green')

bars = ax5.bar(range(11,16), future_pred[:,1], color=future_colors, alpha=0.7)
ax5.set_title("Future Congestion Levels", fontweight='bold', fontsize=12)
ax5.set_xlabel("Future Minutes", fontsize=10)
ax5.set_ylabel("Vehicle Count", fontsize=10)
ax5.set_xticks(range(11,16))
ax5.set_xticklabels([f"T+{i}\n{window_times_12h[i+9][:5]}" for i in range(1,6)], rotation=45, fontsize=8)

# Add congestion labels with adjusted positions
for i, (bar, level) in enumerate(zip(bars, congestion_levels)):
    short = level.split()[0]
    ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height()/2,
             short, ha='center', va='center', color='white', fontweight='bold', fontsize=8)

# [6] Trend from Past to Future
ax6 = plt.subplot(2,3,6)
trend_data = np.concatenate([past_10[-3:,1], future_pred[:,1]])
trend_labels = ['T-2', 'T-1', 'NOW', 'T+1', 'T+2', 'T+3', 'T+4', 'T+5']
colors = ['blue', 'blue', 'green'] + ['red']*5

# Plot lines with markers
for i in range(len(trend_data)-1):
    ax6.plot([i, i+1], [trend_data[i], trend_data[i+1]], 
             color=colors[i+1], linewidth=2, marker='o', markersize=8,
             markerfacecolor=colors[i+1])

ax6.set_title("Live Traffic Trend", fontweight='bold', fontsize=12)
ax6.set_xlabel("Time Relative to Now", fontsize=10)
ax6.set_ylabel("Vehicle Count", fontsize=10)
ax6.grid(True, alpha=0.3)
ax6.set_xticks(range(8))
ax6.set_xticklabels(trend_labels, fontsize=9)

# Add value labels with adjusted positions
for i, (label, value) in enumerate(zip(trend_labels, trend_data)):
    offset = 5 if value < np.max(trend_data) * 0.8 else -15
    ax6.text(i, value + offset, f'{int(value)}', 
             ha='center', va='bottom' if offset > 0 else 'top', 
             fontsize=9, fontweight='bold')

# Add timestamp with adjusted position
plt.figtext(0.5, 0.02, 
            f"Generated: {datetime.now().strftime('%A, %B %d, %Y at %I:%M:%S %p')}\n"
            f"Based on Current Time: {window_times_12h[9]} | Total Runtime: {(time.time() - script_start_timestamp):.2f}s", 
            ha='center', fontsize=9, style='italic', 
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

# Adjust layout with more space for title and timestamp
plt.subplots_adjust(top=0.88, bottom=0.12, hspace=0.3, wspace=0.25)
plt.show()

plot_end = time.time()
plot_duration = plot_end - plot_start
print(f"✓ Plot generation time: {plot_duration:.2f} seconds")

# ============================================================
# 11. SAVE RESULTS
# ============================================================
print("\n" + "-"*60)
print("STEP 9: SAVING RESULTS")
print("-"*60)

save_start = time.time()

try:
    # Create directory
    if not os.path.exists('predictions'):
        os.makedirs('predictions')
    
    # Generate filename with current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save current prediction
    results_df = pd.DataFrame({
        'Prediction_Time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Current_Time': window_times_12h[9],
        'Current_Date': window_dates[9],
        'Current_Vehicle_Count': current_count,
        'Current_Speed': past_10[-1][0],
        'Current_Wait': past_10[-1][2],
        'T+1_Time': window_times_12h[10],
        'T+1_Count': future_pred[0][1],
        'T+2_Time': window_times_12h[11],
        'T+2_Count': future_pred[1][1],
        'T+3_Time': window_times_12h[12],
        'T+3_Count': future_pred[2][1],
        'T+4_Time': window_times_12h[13],
        'T+4_Count': future_pred[3][1],
        'T+5_Time': window_times_12h[14],
        'T+5_Count': future_pred[4][1],
        'Peak_Count': max_future_count,
        'Peak_Time': window_times_12h[peak_idx + 10],
        'Average_Future_Count': avg_future_count,
        'Change_Percent': ((avg_future_count - avg_past_count)/avg_past_count*100)
    }, index=[0])
    
    filename = f'predictions/current_time_prediction_{timestamp}.csv'
    results_df.to_csv(filename, index=False)
    print(f"✓ Results saved to: {filename}")
    
    # Also save plot
    plt.figure(figsize=(16, 10))
    # (Plot code would be here to save automatically)
    # plt.savefig(f'predictions/traffic_plot_{timestamp}.png', dpi=150, bbox_inches='tight')
    
except Exception as e:
    print(f"✗ Could not save results: {e}")

save_end = time.time()
save_duration = save_end - save_start
print(f"Saving time: {save_duration:.2f} seconds")

# ============================================================
# 12. FINAL SUMMARY
# ============================================================
script_end_time = datetime.now()
total_runtime = time.time() - script_start_timestamp

print("\n" + "="*70)
print("STEP 10: FINAL EXECUTION SUMMARY")
print("="*70)
print(f"""
TIMELINE:
   • Script Started:  {script_start_time.strftime('%I:%M:%S %p')}
   • Current Time:     {window_times_12h[9]}
   • Script Ended:     {script_end_time.strftime('%I:%M:%S %p')}
   • Total Runtime:    {total_runtime:.2f} seconds

CURRENT TRAFFIC STATUS:
   • Time: {window_times_12h[9]}
   • Vehicles: {current_count:.0f}
   • Speed: {past_10[-1][0]:.1f} km/h
   • Wait: {past_10[-1][2]:.1f}s

NEXT 5 MINUTES FORECAST:
   • T+1 ({window_times_12h[10]}): {future_pred[0][1]:.0f} vehicles
   • T+2 ({window_times_12h[11]}): {future_pred[1][1]:.0f} vehicles
   • T+3 ({window_times_12h[12]}): {future_pred[2][1]:.0f} vehicles
   • T+4 ({window_times_12h[13]}): {future_pred[3][1]:.0f} vehicles
   • T+5 ({window_times_12h[14]}): {future_pred[4][1]:.0f} vehicles

PEAK PREDICTION:
   • Peak at: {window_times_12h[peak_idx + 10]} ({peak_idx + 1} min from now)
   • Peak count: {max_future_count:.0f} vehicles
""")
print("="*70)
print("✓ PREDICTION COMPLETED - BASED ON CURRENT TIME")
print("="*70)