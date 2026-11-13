# iot-energy-anomaly
Industrial IoT Energy Anomaly Detection & Maintenance Analytics

An end-to-end AI/ML pipeline for industrial IoT telemetry analysis, designed to detect abnormal machine behavior, quantify excess energy consumption, estimate financial loss, and generate actionable maintenance recommendations.

The project simulates telemetry from three industrial assets:

Chiller-01 — Central Cooling Plant

Compressor-02 — Compressed Air System

Pump-03 — Cooling Water Circulation

The system covers the complete workflow:

Synthetic IoT Data → Data Cleaning → Feature Engineering → Anomaly Detection → Energy Waste Analysis → Root-Cause Diagnostics → FastAPI Service

1. Project Objective

Industrial machines continuously generate telemetry such as power consumption, flow rate, temperature, pressure, and operational status.

The objective of this project is to build a lightweight production-oriented analytics system that can:

Generate realistic industrial telemetry.

Inject real-world sensor problems and equipment degradation.

Clean and validate the telemetry.

Engineer domain-specific efficiency features.

Detect anomalous machine behavior using Machine Learning.

Assign operational severity levels.

Estimate excess energy consumption.

Estimate financial energy loss.

Generate maintenance-oriented diagnostic messages.

Expose the trained model through a REST API.

The assignment specifically requires at least 7 days of 1-minute telemetry for each machine, equivalent to 10,080 readings per machine.

2. System Architecture

                  ┌──────────────────────────┐
                  │   Synthetic Telemetry    │
                  │    generate_data.py      │
                  └────────────┬─────────────┘
                               │
                               ▼
                  ┌──────────────────────────┐
                  │   Raw Telemetry CSV      │
                  │ data/raw_telemetry.csv   │
                  └────────────┬─────────────┘
                               │
                               ▼
                  ┌──────────────────────────┐
                  │ Data Cleaning &          │
                  │ Feature Engineering      │
                  │ data_processor.py        │
                  └────────────┬─────────────┘
                               │
                               ▼
                  ┌──────────────────────────┐
                  │ Cleaned Telemetry        │
                  │ data/cleaned_telemetry   │
                  └────────────┬─────────────┘
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
          ┌──────────────────┐   ┌──────────────────┐
          │ Visualization     │   │ ML Training      │
          │ visualize.py      │   │ train.py         │
          └──────────────────┘   └────────┬─────────┘
                                          │
                                          ▼
                               ┌─────────────────────┐
                               │ Anomaly Detector    │
                               │ + Scaler            │
                               └──────────┬──────────┘
                                          │
                                          ▼
                               ┌─────────────────────┐
                               │ Insight Generation   │
                               │ insights.py          │
                               └──────────┬──────────┘
                                          │
                              ┌───────────┴───────────┐
                              ▼                       ▼
                    Energy Waste Analysis       Diagnostics
                              │                       │
                              └───────────┬───────────┘
                                          ▼
                               ┌─────────────────────┐
                               │      FastAPI         │
                               │       main.py        │
                               └──────────┬──────────┘
                                          │
                                          ▼
                               ┌─────────────────────┐
                               │ REST API / Dashboard │
                               │ Maintenance Team     │
                               └─────────────────────┘

3. Repository Structure

iot-energy-anomaly/
│
├── data/
│   ├── raw_telemetry.csv
│   └── cleaned_telemetry.csv
│
├── models/
│   ├── anomaly_detector.joblib
│   └── scaler.joblib
│
├── outputs/
│   ├── trends.png
│   ├── efficiency_correlation.png
│   └── anomalies_detected.png
│
├── src/
│   ├── __init__.py
│   ├── generate_data.py
│   ├── data_processor.py
│   ├── train.py
│   ├── visualize.py
│   ├── insights.py
│   └── main.py
│
├── tests/
│   ├── __init__.py
│   ├── test_data_processor.py
│   └── test_api.py
│
├── .gitignore
├── Dockerfile
├── docker-compose.yaml
├── requirements.txt
├── README.md
└── run.sh

4. Telemetry Dataset

The synthetic dataset contains minute-level telemetry for the three industrial assets.

Input Columns

Column

Description

timestamp

ISO-8601 telemetry timestamp

asset_id

Machine identifier

power_kw

Electrical power consumption

flow_rate

Machine output flow

supply_temp_c

Supply/outlet temperature

return_temp_c

Return/inlet temperature

pressure_bar

Discharge pressure

status

RUNNING, STANDBY, or OFF

These fields follow the assignment's required telemetry schema.

5. Realistic Data Problems

The data generator intentionally introduces realistic industrial telemetry problems.

Sensor glitches

Communication dropouts create missing timestamps and NaN periods.

Physical outliers

Examples include:

power_kw < 0
temperature = 999°C

These values represent physically impossible or corrupted sensor readings.

Frozen sensors

Temperature values can remain unchanged for an extended period while the machine is running.

Operational degradation

The system simulates gradual deterioration by increasing power consumption by approximately 20–30% over 48 hours for the same flow condition.

This represents conditions such as:

Heat exchanger fouling

Compressor seal wear

Reduced thermal efficiency

These injected conditions are part of the assignment's required anomaly scenarios.

6. Data Cleaning

src/data_processor.py performs the preprocessing pipeline.

Missing-value handling

Time-series gaps are handled using appropriate techniques such as:

Forward filling

Backward filling

Interpolation

Physical validation

Invalid sensor readings are identified and removed or corrected according to physical constraints.

For example:

power_kw >= 0
temperature ∈ [-5°C, 110°C]

Frozen sensor detection

A rolling standard deviation is used to identify sensor freezes:

rolling_std = temperature.rolling(window=30).std()
frozen_sensor = rolling_std == 0

Output

data/cleaned_telemetry.csv

The cleaning requirements and rolling-window freeze detection follow the assignment specification.

7. Feature Engineering

The pipeline creates domain-specific features that describe machine efficiency and operational behavior.

Delta Temperature

delta_t = return_temp_c - supply_temp_c

This represents the temperature differential across the equipment.

Specific Energy

specific_energy =
    power_kw / (flow_rate + 1e-5)

This measures energy consumption relative to machine output.

Rolling Power

rolling_power_mean_1h

One-hour rolling average of power consumption.

Load Ratio

load_ratio =
    current_power / rated_nominal_capacity

These engineered features are designed to make abnormal energy behavior easier to identify.

8. Exploratory Data Analysis

The visualization module automatically generates diagnostic charts.

Run:

python src/visualize.py

Generated outputs:

outputs/
├── trends.png
├── efficiency_correlation.png
└── anomalies_detected.png

1. Time-Series Trend

Displays:

Power consumption

Flow rate

Normal periods

Abnormal periods

2. Efficiency Correlation

Shows the relationship between:

Power (kW)
        vs
Flow Rate

with operational status information.

3. Anomaly Detection Timeline

Highlights detected abnormal intervals on the telemetry timeline.

These are the three visualization categories requested in the assignment.

9. Machine Learning

Anomaly Detection

The project uses an unsupervised anomaly detection approach because the telemetry does not require manually labelled anomaly classes.

The model is trained primarily using:

status == RUNNING

This prevents normal STANDBY and OFF operating states from being incorrectly treated as machine anomalies.

Model

The anomaly detection model is implemented using a scikit-learn unsupervised algorithm.

Example:

Isolation Forest

The training pipeline also uses feature scaling where appropriate.

10. Anomaly Score

The raw model output is transformed into a normalized score:

0.0 → 1.0

where higher values indicate more abnormal behavior.

Operational Classification

Anomaly Score

Classification

< 0.50

NORMAL

0.50 – < 0.75

WARNING

>= 0.75

CRITICAL

This allows the ML output to be translated into operational severity levels for maintenance teams.

11. Model Serialization

After training, the model and scaler are serialized to disk.

Example:

models/
├── anomaly_detector.joblib
└── scaler.joblib

The API loads these artifacts during startup instead of retraining the model for every request.

12. Training Pipeline

Run:

python src/train.py

The training pipeline:

Loads cleaned telemetry.

Selects RUNNING records.

Selects ML features.

Applies feature scaling.

Trains the anomaly detector.

Calculates anomaly scores.

Converts scores to:

NORMAL

WARNING

CRITICAL

Saves the model.

Stores the scaler.

Produces evaluation information.

13. Evaluation

The model is evaluated against the intentionally injected abnormal conditions.

The main anomaly scenarios are:

Power Surge

Abnormally high power consumption without a proportional increase in flow.

Frozen Sensor

Temperature remains unchanged for an extended period while the machine is running.

Efficiency Drift

Specific energy gradually increases over time.

The evaluation focuses not only on model performance but also on operational usefulness.

14. Precision vs Recall

Industrial anomaly detection requires a balance between precision and recall.

A false positive can trigger:

Unnecessary maintenance

Production interruption

Technician investigation

Operational cost

A false negative can be more serious because a genuine machine problem may remain undetected and potentially result in:

Energy waste

Equipment degradation

Unplanned downtime

Maintenance cost

Therefore, the decision threshold should prioritize reliable detection while controlling unnecessary maintenance alerts.

The assignment specifically requires documenting this precision/recall trade-off and explaining the selected decision threshold.

15. Energy Waste Calculation

The system estimates how much additional energy was consumed during anomalous operation.

Baseline

For each operating condition, the system estimates expected power consumption based on normal behavior for the corresponding flow/operating condition.

Excess Energy

Excess kWh =
Σ (Actual kW - Baseline kW) × Δt(minutes) / 60

Financial Loss

The estimated financial loss is calculated using:

Electricity tariff = $0.12 / kWh

Therefore:

Estimated Cost Loss =
Excess kWh × 0.12

The assignment specifies this calculation and tariff.

16. Root-Cause Diagnostics

The system converts telemetry anomalies into actionable maintenance recommendations.

Example: Power Surge

High power surge detected without proportional flow increase.
Check electrical motor load or compressor stage bypass.

Example: Frozen Temperature Sensor

Sensor freeze detected on temperature probe.
Possible thermocouple failure or communication loop lockup.

Example: Gradual Efficiency Loss

Gradual thermal efficiency loss detected.
Recommend inspecting heat exchanger tubes for fouling or refrigerant leak.

The goal is to move from:

"Anomaly detected"

to:

"What happened?"
        ↓
"How serious is it?"
        ↓
"How much energy was wasted?"
        ↓
"What should maintenance inspect?"

The diagnostic heuristics follow the scenarios specified in the assignment.

17. Optional LLM Operator Assistant

The diagnostic output can optionally be passed to an LLM such as:

OpenAI

Anthropic Claude

Local Ollama

The LLM can convert the diagnostic results into a concise maintenance shift-handoff report.

Example:

Shift Handoff – Chiller-01

• Severity: CRITICAL
• Issue: Gradual efficiency degradation
• Excess Energy: 42.7 kWh
• Estimated Loss: $5.12
• Recommended Action:
  Inspect heat exchanger for fouling and verify refrigerant conditions.

This component is optional and is not required for the core anomaly-detection pipeline.

18. FastAPI Service

The trained model is exposed through a lightweight FastAPI application.

Start the API:

uvicorn src.main:app --reload

API:

http://127.0.0.1:8000

Swagger documentation:

http://127.0.0.1:8000/docs

19. API Endpoints

Health Check

GET /health

Purpose:

Verify API availability

Verify model loaded successfully

Example response:

{
    "status": "healthy",
    "model_loaded": true
}

20. Diagnose Single Telemetry Reading

POST /api/v1/diagnose

Request

{
    "timestamp": "2026-10-01T14:30:00Z",
    "asset_id": "Chiller-01",
    "power_kw": 285.5,
    "flow_rate": 450.0,
    "supply_temp_c": 7.2,
    "return_temp_c": 12.8,
    "pressure_bar": 4.2,
    "status": "RUNNING"
}

The endpoint returns:

Anomaly score

Classification

Diagnostic message

Example:

{
    "asset_id": "Chiller-01",
    "anomaly_score": 0.82,
    "classification": "CRITICAL",
    "diagnostic": "High power surge detected without proportional flow increase."
}

The request structure and required response behavior are defined in the assignment.

21. Batch Audit

POST /api/v1/batch-audit

Accepts:

CSV upload

Batch JSON telemetry

The endpoint analyzes a batch of telemetry and returns:

Total records analyzed
Number of anomalies
Excess energy waste (kWh)
Estimated cost loss ($)
Flagged time intervals

This enables maintenance teams to audit a historical operating period rather than analyzing only one sensor reading.

22. Installation

Clone Repository

git clone <YOUR_GITHUB_REPOSITORY_URL>
cd iot-energy-anomaly

Create Virtual Environment

Windows

python -m venv .venv
.venv\Scripts\activate

Linux/macOS

python3 -m venv .venv
source .venv/bin/activate

Install Dependencies

pip install -r requirements.txt

23. Run the Complete Pipeline

Step 1 — Generate telemetry

python src/generate_data.py

Expected output:

data/raw_telemetry.csv

Step 2 — Clean and engineer features

python src/data_processor.py

Expected output:

data/cleaned_telemetry.csv

Step 3 — Generate visualizations

python src/visualize.py

Expected outputs:

outputs/trends.png
outputs/efficiency_correlation.png
outputs/anomalies_detected.png

Step 4 — Train anomaly detector

python src/train.py

Expected outputs:

models/anomaly_detector.joblib
models/scaler.joblib

Step 5 — Start API

uvicorn src.main:app --reload

24. Docker

The project also supports containerized execution.

Build:

docker compose build

Run:

docker compose up

The service will be available at:

http://localhost:8000

The assignment requires a reproducible environment using a requirements file and a Dockerfile/compose setup or equivalent single-command execution.

25. Testing

Run the test suite:

pytest tests/ -v

The tests cover:

Data preprocessing

Feature engineering

API behavior

/diagnose endpoint

Example:

tests/
├── test_data_processor.py
└── test_api.py

26. Example End-to-End Workflow

1. Generate synthetic industrial telemetry
              ↓
2. Inject sensor and operational anomalies
              ↓
3. Clean invalid/missing/frozen data
              ↓
4. Engineer efficiency features
              ↓
5. Train anomaly detection model
              ↓
6. Calculate anomaly scores
              ↓
7. Classify:
      NORMAL / WARNING / CRITICAL
              ↓
8. Calculate baseline power
              ↓
9. Calculate excess kWh
              ↓
10. Estimate financial loss
              ↓
11. Generate diagnostic recommendation
              ↓
12. Serve predictions through FastAPI

27. Key Engineering Decisions

Why unsupervised anomaly detection?

The system does not depend on manually labelled industrial failures. Instead, it learns normal operating behavior and identifies deviations.

Why train primarily on RUNNING data?

OFF and STANDBY naturally have different power and flow characteristics. Including them as normal training examples can distort the definition of machine normality and produce misleading anomaly scores.

Why domain features?

Raw power values alone are insufficient.

For example:

Power = 250 kW

does not necessarily indicate a problem.

But:

Power = 250 kW
Flow = unchanged
Specific Energy = increasing

provides stronger evidence of efficiency degradation.

Why convert scores into severity levels?

Maintenance engineers generally need an operational decision rather than an abstract ML score.

0.31 → NORMAL
0.63 → WARNING
0.88 → CRITICAL

28. Production Considerations

For production deployment, this architecture could be extended with:

MQTT/Kafka telemetry ingestion

TimescaleDB or another time-series database

Real-time streaming inference

Model monitoring

Drift detection

Alert management

Role-based access control

Authentication and authorization

Centralized logging

Prometheus/Grafana monitoring

Cloud deployment

LLM-assisted maintenance reporting

These extensions are outside the core assignment scope.

29. Future Improvements

Potential improvements include:

Online/streaming anomaly detection

Asset-specific models

Automated model retraining

SHAP-based anomaly explanations

Remaining Useful Life (RUL) prediction

Predictive maintenance modeling

Real-time alert notifications

MQTT integration

Grafana dashboard

Time-series database integration

LLM maintenance assistant

30. Project Deliverables

Component

Status

Synthetic telemetry generation

✅

7+ days minute-level data

✅

Missing data injection

✅

Physical outlier injection

✅

Sensor freeze simulation

✅

Operational degradation simulation

✅

Data cleaning

✅

Feature engineering

✅

Exploratory visualization

✅

ML anomaly detection

✅

Anomaly scoring

✅

NORMAL/WARNING/CRITICAL classification

✅

Model serialization

✅

Energy waste calculation

✅

Cost-loss estimation

✅

Diagnostic heuristics

✅

FastAPI service

✅

Batch audit endpoint

✅

Health endpoint

✅

Automated tests

✅

Docker support

✅

31. Conclusion

This project demonstrates an end-to-end approach to industrial IoT energy and maintenance analytics.

Rather than simply detecting anomalies, the system connects machine-learning predictions with operational decisions:

Telemetry
   ↓
Data Quality
   ↓
Machine Learning
   ↓
Anomaly Detection
   ↓
Energy Waste
   ↓
Financial Impact
   ↓
Root Cause
   ↓
Maintenance Action

The architecture is modular so individual components can be replaced or extended without redesigning the entire pipeline.

Author

Vinoth M

Data Science | Machine Learning | AI/ML Engineering

GitHub: https://github.com/vinothm002

LinkedIn: https://linkedin.com/in/vinoth-m-m-sc/

License

This project was developed as an AI/ML engineering assignment and demonstration project.
