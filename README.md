# 🚦 Traffic Congestion Prediction Using Digital Twin and LSTM

## 📌 Project Description

This project focuses on predicting traffic congestion using **Digital Twin technology** and a **Long Short-Term Memory (LSTM)** deep learning model.

The system creates a virtual representation of a real-world traffic environment using **SUMO (Simulation of Urban Mobility)**. Traffic information such as vehicle count, average speed, and waiting time is collected from the simulation and used to train the LSTM model.

The trained model can predict future traffic conditions and help improve traffic management and smart transportation systems.

## 🎯 Objectives

* Simulate real-world traffic using Digital Twin technology.
* Generate and collect traffic data using SUMO.
* Analyze traffic parameters such as vehicle count, speed, and waiting time.
* Train an LSTM model for traffic congestion prediction.
* Predict future traffic conditions.
* Support intelligent traffic management.

## 🛠️ Technologies Used

* **Python**
* **LSTM (Long Short-Term Memory)**
* **TensorFlow / Keras**
* **SUMO**
* **TraCI**
* **Pandas**
* **NumPy**
* **Flask**
* **Matplotlib**
* **Scikit-learn**

## 📊 Dataset

The traffic dataset contains the following attributes:

| Feature         | Description                 |
| --------------- | --------------------------- |
| Date            | Date of traffic observation |
| Time_AM_PM      | Time period                 |
| Simulation_Step | Simulation time step        |
| Vehicle_Count   | Number of vehicles          |
| Average_Speed   | Average vehicle speed       |
| Waiting_Time    | Average waiting time        |

## 🧠 Machine Learning Model

The project uses an **LSTM (Long Short-Term Memory)** neural network because traffic conditions are time-dependent.

The model learns patterns from historical traffic data and predicts future traffic conditions.

### LSTM Architecture

```text
Input Traffic Data
       ↓
LSTM Layer (64 units)
       ↓
LSTM Layer (32 units)
       ↓
Dense Layer
       ↓
Traffic Prediction
```

## 🌐 Digital Twin Environment

**SUMO (Simulation of Urban Mobility)** is used to create the virtual traffic environment.

The Digital Twin represents traffic conditions in a simulated environment and provides data that can be analyzed by the LSTM model.

```text
Real Traffic Environment
          ↓
     Digital Twin
          ↓
        SUMO
          ↓
    Traffic Data
          ↓
     LSTM Model
          ↓
   Traffic Prediction
```

## 📁 Project Structure

```text
DigitalTwin_SUMO/
│
├── data/
│   └── traffic_data.csv
│
├── model/
│   └── lstm_model.keras
│
├── src/
│   ├── train.py
│   ├── predict.py
│   └── app.py
│
├── SUMO/
│   ├── traffic_simulation.sumocfg
│   └── other SUMO files
│
├── requirements.txt
│
└── README.md
```

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/DigitalTwin_SUMO.git
```

### 2. Open the project

```bash
cd DigitalTwin_SUMO
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Install SUMO

Download and install SUMO on your system.

Make sure SUMO is added to the system PATH.

## ▶️ How to Run

### Run the traffic simulation

```bash
python src/traffic_simulation.py
```

### Train the LSTM model

```bash
python src/train.py
```

### Run prediction

```bash
python src/predict.py
```

### Run Flask application

```bash
python src/app.py
```

Then open the application in your browser.

```text
http://127.0.0.1:5000
```

## 📈 Project Workflow

```text
SUMO Traffic Simulation
          ↓
   Data Collection
          ↓
    Data Preprocessing
          ↓
    Feature Selection
          ↓
    LSTM Model Training
          ↓
     Model Evaluation
          ↓
   Traffic Prediction
          ↓
   Congestion Analysis
```

## 🔮 Future Enhancements

* Real-time traffic prediction.
* Integration with live traffic data.
* Traffic signal optimization.
* Real-time dashboard.
* Integration with IoT sensors.
* Improved deep learning models.
* Deployment on cloud platforms.

