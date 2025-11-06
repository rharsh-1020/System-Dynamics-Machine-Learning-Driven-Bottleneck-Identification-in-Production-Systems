# System-Dynamics-Machine-Learning-Driven-Bottleneck-Identification-in-Production-Systems


---

## Overview

This repository implements a **hybrid System Dynamics (SD)** and **Machine Learning (ML)** framework to **predict** and **mitigate bottlenecks** in serial production systems.  
It models how **bottlenecks shift dynamically** across stations and demonstrates how **data-driven mitigation** can improve throughput and reduce queue congestion.

**Core idea:**  
> Learn failure risk from real manufacturing data (SECOM), map it to capacity variations in a simulated line, and evaluate proactive mitigation strategies using System Dynamics and Discrete-Event Simulation (SimPy).

---

## ⚙️ Methodology Summary

| Phase | Description | Key Outcome |
|-------|--------------|--------------|
| **Phase 1 – SD & DES** | Model a 3-station line in Vensim/Python; simulate a mid-run capacity shift (S2 → S3). | Throughput ≈ 262 units; bottleneck migration verified. |
| **Phase 2 – ML Baseline (SECOM)** | Random Forest on 1,567 × 590 sensor features (positive rate 6.64 %). | ROC-AUC = 0.788, PR-AUC = 0.200 (F1 thr ≈ 0.10). |
| **Phase 3 – Integration** | Convert predicted risk → capacity schedule → SimPy simulation. | Mitigation ↑ throughput +24.5 % vs baseline; +129.7 % vs risk. |
| **Phase 4 add-on – GRU** | Train GRU on short SECOM sequences (T = 10). | Validates sequential risk pipeline (ROC ≈ 0.43). |
| **Phase 5 – ST-GNN-lite (optional)** | Graph + temporal model on sim queues for direct bottleneck classification. | Accuracy 0.52 · Macro-F1 0.34. |

---

## Environment Setup

```bash
conda create -n bneck python=3.11 -y
conda activate bneck
pip install -r requirements.txt
Key dependencies:
numpy · pandas · matplotlib · scikit-learn · pyyaml · simpy · xgboost · tensorflow / keras

## **Data Source**
Dataset: SECOM Data Set — UCI Machine Learning Repository

kotlin
Copy code
data/
├── secom.data
└── secom_labels.data
First column = {−1,+1} → map to {0, 1} for binary classification.

Positive rate ≈ 6.6 %.

Do not commit raw data if restricted by license.

## How to Run
1️⃣ Train the ML Baseline
bash
Copy code
python models/01_secom_baseline.py
Generates ROC/PR curves, confusion matrix, and feature importances.

2️⃣ Run Simulations
bash
Copy code
python sim/line_sim_schedule.py --tag baseline
python sim/line_sim_schedule.py --tag risk --events sim/risk_schedule.csv
python sim/line_sim_schedule.py --tag mitigate --events sim/mitigation_schedule.csv
Outputs JSON/CSV summaries and plots of throughput, queues, and utilization.

3️⃣ (Optional) Train GRU Model
bash
Copy code
python models/02_secom_gru.py
4️⃣ (Optional) ST-GNN Demo
bash
Copy code
python models/03_stgnn_demo.py
📈 Key Results (From Report)
Scenario	Throughput	Δ vs Baseline	Util (S1/S2/S3 %)	Avg Queue S2
Baseline	286	—	99.8 / 99.7 / 59.6	94.9
Risk	155	−45.8 %	99.8 / 99.3 / 32.3	160.6
Mitigate	356	+24.5 %	99.8 / 99.6 / 74.2	60.3

## Takeaway:

The ML-triggered capacity boost significantly restores flow and reduces queues at the active constraint.

💬 Discussion & Insights
Dynamic bottleneck observed (S2 → S3 migration over time).

ML risk mapping allows proactive capacity boosting during high-risk windows.

System Dynamics + ML integration demonstrates explainable, data-driven control of production lines.

## Limitations & Future Work
Align SECOM sensor features to station IDs for station-level prediction.

Extend DES/SD to parallel and merge network topologies.

Replace RF with GNN or RL-based agents for adaptive capacity control.

Study real-time deployment in industrial IoT contexts.
