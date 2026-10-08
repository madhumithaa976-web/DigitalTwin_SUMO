import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import math

print("="*60)
print("TRAFFIC PREDICTION - TRAINING PHASE WITH MSE, RMSE & MAE")
print("="*60)

# -----------------------------
# 1. Load Data (With Date & Time)
# -----------------------------
csv_path = "C:/Users/MADHUMITHA A/OneDrive/Desktop/DigitalTwin_SUMO/module3_preprocessing/processed_data.csv"
df = pd.read_csv(csv_path)

print(f"\n📁 CSV Loaded Successfully")
print(f"📊 Total Rows: {len(df)}")
print(f"📋 Columns: {list(df.columns)}")

# Combine Date + Time into single datetime column
df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"])

# Sort by datetime (Important for time series)
df = df.sort_values("Datetime")

print("✅ Date & Time processed correctly!")

# -----------------------------
# 2. Select Features (Model input)
# -----------------------------
features = ["Average_Speed", "Vehicle_Count", "Waiting_Time"]
data = df[features].values

print(f"\n✅ Using Features: {features}")
print(f"✅ Data Shape: {data.shape}")

# -----------------------------
# 3. Scaling
# -----------------------------
print("\n🔄 Scaling Data...")
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)
print("✅ Scaling Done!")

# -----------------------------
# 4. Create Sequences (10 → 5)
# -----------------------------
print("\n🔄 Creating Sequences (Past 10 → Future 5)...")

X, y = [], []

for i in range(10, len(scaled_data) - 5):
    X.append(scaled_data[i-10:i])
    y.append(scaled_data[i:i+5])

X = np.array(X)
y = np.array(y)

# Flatten output (5 × 3 = 15)
y = y.reshape(y.shape[0], 15)

print(f"✅ Input Shape: {X.shape}")
print(f"✅ Output Shape: {y.shape}")

# -----------------------------
# 5. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

print(f"\n📚 Training Samples: {len(X_train)}")
print(f"🧪 Testing Samples: {len(X_test)}")

# -----------------------------
# 6. Build LSTM Model
# -----------------------------
print("\n🏗️ Building LSTM Model...")

model = Sequential([
    Input(shape=(10, 3)),
    LSTM(64, return_sequences=False),
    Dense(32, activation='relu'),
    Dense(15)
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

print("✅ Model Built Successfully!")

# -----------------------------
# 7. Early Stopping
# -----------------------------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# -----------------------------
# 8. Train Model
# -----------------------------
print("\n🚀 Training Started...")

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

print("\n✅ Training Completed!")

# -----------------------------
# 9. Calculate MSE, RMSE and MAE
# -----------------------------
print("\n📊 Calculating Performance Metrics...")

# Make predictions on test set
y_pred = model.predict(X_test, verbose=0)

# Reshape back to original format for easier interpretation
y_test_reshaped = y_test.reshape(y_test.shape[0], 5, 3)
y_pred_reshaped = y_pred.reshape(y_pred.shape[0], 5, 3)

# Calculate metrics for each feature
print("\n" + "="*60)
print("PERFORMANCE METRICS BY FEATURE")
print("="*60)

feature_metrics = {}

for i, feature in enumerate(features):
    # Extract the feature values for all time steps
    y_test_feature = y_test_reshaped[:, :, i].flatten()
    y_pred_feature = y_pred_reshaped[:, :, i].flatten()
    
    # Calculate metrics
    mse = mean_squared_error(y_test_feature, y_pred_feature)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_test_feature, y_pred_feature)
    
    # Store metrics
    feature_metrics[feature] = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae
    }
    
    print(f"\n📊 {feature}:")
    print(f"   MSE : {mse:.6f}")
    print(f"   RMSE: {rmse:.6f}")
    print(f"   MAE : {mae:.6f}")

# Calculate overall metrics (all features combined)
y_test_flat = y_test.flatten()
y_pred_flat = y_pred.flatten()

overall_mse = mean_squared_error(y_test_flat, y_pred_flat)
overall_rmse = math.sqrt(overall_mse)
overall_mae = mean_absolute_error(y_test_flat, y_pred_flat)

print("\n" + "="*60)
print("OVERALL PERFORMANCE METRICS")
print("="*60)
print(f"\n📊 Overall (All Features Combined):")
print(f"   MSE : {overall_mse:.6f}")
print(f"   RMSE: {overall_rmse:.6f}")
print(f"   MAE : {overall_mae:.6f}")

# Calculate metrics for each prediction minute
print("\n" + "="*60)
print("PERFORMANCE METRICS BY MINUTE")
print("="*60)

minute_metrics = []
for minute in range(5):
    minute_mse_list = []
    minute_mae_list = []
    
    for feature_idx in range(3):
        y_test_minute = y_test_reshaped[:, minute, feature_idx]
        y_pred_minute = y_pred_reshaped[:, minute, feature_idx]
        
        minute_mse_list.append(mean_squared_error(y_test_minute, y_pred_minute))
        minute_mae_list.append(mean_absolute_error(y_test_minute, y_pred_minute))
    
    avg_minute_mse = np.mean(minute_mse_list)
    avg_minute_rmse = math.sqrt(avg_minute_mse)
    avg_minute_mae = np.mean(minute_mae_list)
    
    minute_metrics.append({
        'minute': minute + 1,
        'MSE': avg_minute_mse,
        'RMSE': avg_minute_rmse,
        'MAE': avg_minute_mae
    })
    
    print(f"\n📊 Minute {minute+1} (Average across features):")
    print(f"   MSE : {avg_minute_mse:.6f}")
    print(f"   RMSE: {avg_minute_rmse:.6f}")
    print(f"   MAE : {avg_minute_mae:.6f}")

# -----------------------------
# 10. Save Model & Scaler
# -----------------------------
print("\n" + "="*60)
print("💾 Saving Model & Scaler...")
print("="*60)

model.save("traffic_model.keras")
joblib.dump(scaler, "scaler.pkl")

# Save metrics to file
metrics_summary = {
    'overall': {
        'MSE': overall_mse,
        'RMSE': overall_rmse,
        'MAE': overall_mae
    },
    'by_feature': feature_metrics,
    'by_minute': minute_metrics
}

# Save metrics as JSON for later reference
import json
with open('training_metrics.json', 'w') as f:
    json.dump(metrics_summary, f, indent=4)

print("\n✅ Model saved as: traffic_model.keras")
print("✅ Scaler saved as: scaler.pkl")
print("✅ Metrics saved as: training_metrics.json")

# -----------------------------
# 11. Plot Training Graph with Metrics
# -----------------------------
print("\n" + "="*60)
print("📈 Generating Training Plots...")
print("="*60)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))  # Changed to 2x3 grid

# Plot 1: Training Loss (MSE)
axes[0,0].plot(history.history['loss'], label='Training Loss', linewidth=2)
axes[0,0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[0,0].set_title(f'Model Loss (MSE) Over Epochs\nFinal Loss: {history.history["val_loss"][-1]:.4f}', fontsize=12, fontweight='bold')
axes[0,0].set_xlabel('Epoch')
axes[0,0].set_ylabel('Loss (MSE)')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# Plot 2: Training MAE
if 'mae' in history.history:
    axes[0,1].plot(history.history['mae'], label='Training MAE', linewidth=2)
    axes[0,1].plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
    axes[0,1].set_title('Model MAE Over Epochs', fontsize=12, fontweight='bold')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('MAE')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)

# Plot 3: MSE by Feature
features_list = list(feature_metrics.keys())
mse_values = [feature_metrics[f]['MSE'] for f in features_list]
bars1 = axes[0,2].bar(features_list, mse_values, color=['#667eea', '#48bb78', '#ed8936'])
axes[0,2].set_title('MSE by Feature', fontsize=12, fontweight='bold')
axes[0,2].set_ylabel('MSE')
axes[0,2].grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, val in zip(bars1, mse_values):
    height = bar.get_height()
    axes[0,2].text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 4: RMSE by Feature
rmse_values = [feature_metrics[f]['RMSE'] for f in features_list]
bars2 = axes[1,0].bar(features_list, rmse_values, color=['#9f7aea', '#f687b3', '#fc8181'])
axes[1,0].set_title('RMSE by Feature', fontsize=12, fontweight='bold')
axes[1,0].set_ylabel('RMSE')
axes[1,0].grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, val in zip(bars2, rmse_values):
    height = bar.get_height()
    axes[1,0].text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 5: MAE by Feature
mae_values = [feature_metrics[f]['MAE'] for f in features_list]
bars3 = axes[1,1].bar(features_list, mae_values, color=['#f6ad55', '#68d391', '#4fd1c5'])
axes[1,1].set_title('MAE by Feature', fontsize=12, fontweight='bold')
axes[1,1].set_ylabel('MAE')
axes[1,1].grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, val in zip(bars3, mae_values):
    height = bar.get_height()
    axes[1,1].text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 6: Metrics by Minute
minutes = [m['minute'] for m in minute_metrics]
mse_by_minute = [m['MSE'] for m in minute_metrics]
rmse_by_minute = [m['RMSE'] for m in minute_metrics]
mae_by_minute = [m['MAE'] for m in minute_metrics]

x = np.arange(len(minutes))
width = 0.25

axes[1,2].bar(x - width, mse_by_minute, width, label='MSE', color='#4299e1')
axes[1,2].bar(x, rmse_by_minute, width, label='RMSE', color='#ed8936')
axes[1,2].bar(x + width, mae_by_minute, width, label='MAE', color='#48bb78')

axes[1,2].set_title('Metrics by Prediction Minute', fontsize=12, fontweight='bold')
axes[1,2].set_xlabel('Minute')
axes[1,2].set_ylabel('Error Value')
axes[1,2].set_xticks(x)
axes[1,2].set_xticklabels([f'Min {m}' for m in minutes])
axes[1,2].legend()
axes[1,2].grid(True, alpha=0.3, axis='y')

plt.suptitle(f'TRAINING RESULTS - Overall MSE: {overall_mse:.4f} | RMSE: {overall_rmse:.4f} | MAE: {overall_mae:.4f}', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('training_results.png', dpi=100, bbox_inches='tight')
plt.show()

# -----------------------------
# 12. Sample Predictions Visualization (3 SAMPLES ONLY)
# -----------------------------
print("\n📊 Generating 3 Sample Predictions...")

# Take exactly 3 samples from test set
n_samples = 3
fig, axes = plt.subplots(n_samples, 3, figsize=(15, 10))

for i in range(n_samples):
    for j, feature in enumerate(features):
        # Get actual and predicted values for this sample
        actual = y_test_reshaped[i, :, j]
        predicted = y_pred_reshaped[i, :, j]
        
        # Plot
        axes[i, j].plot(range(1, 6), actual, 'g-o', label='Actual', linewidth=2.5, markersize=8)
        axes[i, j].plot(range(1, 6), predicted, 'r--s', label='Predicted', linewidth=2.5, markersize=8)
        axes[i, j].set_title(f'Sample {i+1} - {feature}', fontsize=12, fontweight='bold')
        axes[i, j].set_xlabel('Minute', fontsize=10)
        axes[i, j].set_ylabel('Value (scaled)', fontsize=10)
        axes[i, j].legend(loc='upper right', fontsize=9)
        axes[i, j].grid(True, alpha=0.3)
        axes[i, j].set_xticks(range(1, 6))
        
        # Calculate and display metrics for this sample
        sample_mse = mean_squared_error(actual, predicted)
        sample_rmse = math.sqrt(sample_mse)
        sample_mae = mean_absolute_error(actual, predicted)
        
        # Add metrics box
        axes[i, j].text(0.05, 0.95, f'MSE: {sample_mse:.4f}\nRMSE: {sample_rmse:.4f}\nMAE: {sample_mae:.4f}', 
                       transform=axes[i, j].transAxes, fontsize=8,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

plt.suptitle('3 Sample Predictions vs Actual (Test Set)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('3_sample_predictions.png', dpi=100, bbox_inches='tight')
plt.show()

# -----------------------------
# 13. Additional Plot: Combined Metrics Summary
# -----------------------------
print("\n📊 Generating Metrics Summary Plot...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: MSE Comparison
x_pos = np.arange(len(features))
width = 0.35

mse_values = [feature_metrics[f]['MSE'] for f in features]
bars1 = axes[0].bar(x_pos, mse_values, width, color='#4299e1')
axes[0].set_title('Mean Squared Error (MSE) by Feature', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Features')
axes[0].set_ylabel('MSE')
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(features, rotation=45, ha='right')
axes[0].grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars1, mse_values):
    height = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Plot 2: RMSE Comparison
rmse_values = [feature_metrics[f]['RMSE'] for f in features]
bars2 = axes[1].bar(x_pos, rmse_values, width, color='#ed8936')
axes[1].set_title('Root Mean Squared Error (RMSE) by Feature', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Features')
axes[1].set_ylabel('RMSE')
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(features, rotation=45, ha='right')
axes[1].grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars2, rmse_values):
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Plot 3: MAE Comparison
mae_values = [feature_metrics[f]['MAE'] for f in features]
bars3 = axes[2].bar(x_pos, mae_values, width, color='#48bb78')
axes[2].set_title('Mean Absolute Error (MAE) by Feature', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Features')
axes[2].set_ylabel('MAE')
axes[2].set_xticks(x_pos)
axes[2].set_xticklabels(features, rotation=45, ha='right')
axes[2].grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars3, mae_values):
    height = bar.get_height()
    axes[2].text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.suptitle(f'Performance Metrics Comparison - Overall MSE: {overall_mse:.4f}', fontsize=16, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig('metrics_comparison.png', dpi=100, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("🎉 TRAINING COMPLETE WITH DATE & TIME SUPPORT!")
print(f"📊 Overall MSE : {overall_mse:.6f}")
print(f"📊 Overall RMSE: {overall_rmse:.6f}")
print(f"📊 Overall MAE : {overall_mae:.6f}")
print("="*60)

# Print summary
print("\n📋 METRICS SUMMARY")
print("-" * 50)
print(f"{'Feature':<20} {'MSE':<12} {'RMSE':<12} {'MAE':<12}")
print("-" * 50)
for feature in features:
    print(f"{feature:<20} {feature_metrics[feature]['MSE']:<12.6f} {feature_metrics[feature]['RMSE']:<12.6f} {feature_metrics[feature]['MAE']:<12.6f}")
print("-" * 50)
print(f"{'OVERALL':<20} {overall_mse:<12.6f} {overall_rmse:<12.6f} {overall_mae:<12.6f}")
print("="*60)

print("\n📊 FILES GENERATED:")
print("   • traffic_model.keras - Trained LSTM model")
print("   • scaler.pkl - Data scaler")
print("   • training_metrics.json - All metrics in JSON")
print("   • training_results.png - Training plots (MSE, RMSE, MAE)")
print("   • 3_sample_predictions.png - 3 sample predictions with all metrics")
print("   • metrics_comparison.png - MSE, RMSE, MAE comparison by feature")
print("="*60)