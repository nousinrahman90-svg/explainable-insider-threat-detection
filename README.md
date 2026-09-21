# Explainable Machine Learning Framework for Insider Threat and Operational Anomaly Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Research Paper](https://img.shields.io/badge/Paper-Preprint-success.svg)](./Explainable_Insider_Threat_Detection_Nousin_Rahman.pdf)

**Author:** Nousin Rahman  
**Institution:** Department of Information Systems and Statistics, Zicklin School of Business, Baruch College, City University of New York (CUNY)  
**Research Focus:** AI & Machine Learning in Cybersecurity, Enterprise Telemetry, Zero-Trust Infrastructure

---

## Overview

Modern cloud-native enterprise ecosystems generate continuous high-volume log streams spanning authentication manifests, HTTP proxy traffic, and file system modifications. Detecting insider threats and operational anomalies within this data is complicated by extreme class imbalance (<0.1% positive anomalies) and high false-alarm rates that overwhelm Security Operations Centers (SOCs).

This repository provides an end-to-end analytical pipeline implementing:
1. **Behavioral Feature Engineering:** Mathematical formulations for authentication failure ratios, destination host Shannon entropy, normalized data transfer metrics, and after-hours scheduling flags.
2. **Multi-Model Anomaly Benchmarking:** Comparative evaluation of:
   - **Isolation Forest** (Unsupervised boundary isolation)
   - **Deep Neural Autoencoder** (Reconstruction error calibration)
   - **Cost-Sensitive Ensembles (XGBoost / Random Forest)** (Focal positive loss re-weighting)
3. **Operational SOC Metrics:** Evaluation via Precision-Recall AUC (PR-AUC), Detection Recall, F1-score, and False Positive Rate (FPR).
4. **Explainable AI (XAI) Attribution:** Integration of **Tree-SHAP** (SHapley Additive exPlanations) providing local and global forensic justification for flagged alerts.

---

## Repository Structure

```text
├── threat_detection_pipeline.py     # End-to-end Python pipeline (data, models, evaluation, SHAP)
├── Explainable_Insider_Threat_Detection_Nousin_Rahman.pdf  # Full peer-reviewed research manuscript
├── requirements.txt                 # Python dependencies
└── README.md                        # Documentation & setup instructions
```

---

## Quick Start

### 1. Installation

Clone this repository and install dependencies:

```bash
git clone https://github.com/nousinrahman/explainable-insider-threat-detection.git
cd explainable-insider-threat-detection
pip install -r requirements.txt
```

*Recommended `requirements.txt`:*
```text
numpy>=1.22.0
pandas>=1.4.0
scikit-learn>=1.0.0
xgboost>=1.6.0
shap>=0.41.0
matplotlib>=3.5.0
```

### 2. Execute the Detection & Explainability Pipeline

Run the standalone pipeline script:

```bash
python threat_detection_pipeline.py
```

---

## Experimental Benchmark Results

Evaluated against the **Carnegie Mellon University (CMU) CERT Insider Threat Test Dataset (r4.2)**:

| Model Architecture | Precision | Recall | F1-Score | PR-AUC | False Positive Rate (FPR) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Isolation Forest** | 0.384 | 0.692 | 0.494 | 0.512 | 3.82% |
| **One-Class SVM** | 0.312 | 0.641 | 0.420 | 0.448 | 4.95% |
| **Deep Autoencoder** | 0.528 | 0.765 | 0.625 | 0.684 | 2.15% |
| **XGBoost (Cost-Sensitive)** | **0.782** | **0.894** | **0.834** | **0.841** | **0.41%** |

---

## Citation

If you utilize this pipeline or research framework in your academic work, please cite:

```bibtex
@article{rahman2026explainable,
  title={An Explainable Machine Learning Framework for Insider Threat and Anomaly Detection in Cloud-Native Enterprise Infrastructure},
  author={Rahman, Nousin},
  journal={Preprint},
  year={2026},
  institution={Zicklin School of Business, Baruch College, City University of New York}
}
```
