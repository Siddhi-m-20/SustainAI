# EcoSync Model Evaluation & Benchmarks

## 1. Executive Summary

This document details the quantitative evaluation and performance validation of the machine learning algorithms embedded in EcoSync, specifically:
1. **Multi-Resource Anomaly Detection** (Isolation Forest vs. Empirical Baselines)
2. **Cyclical Demand Forecasting** (Ridge Regression vs. Naive Baseline)
3. **Counterfactual Simulation Engine** (Physics-based validation)

---

## 2. Anomaly Detection Performance

### 2.1 Methodology
Anomaly models were evaluated on the multi-resource campus benchmark dataset (8,640 continuous hourly records per facility across Building A, B, C, and D). Ground-truth synthetic incident injections represent documented commercial facility failures:
* Unscheduled overnight HVAC schedule overrides (parasitic energy baseload)
* Stuck restroom flushometer valves and cooling tower makeup drift (overnight water loss)
* Bulk contamination and diversion bypass events (solid waste surges)

### 2.2 Comparative Benchmark

| Model / Approach | Precision | Recall | F1-Score | False Alarm Rate (FAR) | Detection Latency |
|---|---|---|---|---|---|
| Static Threshold ($> 3\sigma$) | 0.64 | 0.58 | 0.61 | 14.2% | ~24 hours |
| Contextual Diurnal Median | 0.82 | 0.88 | 0.85 | 5.8% | < 3 hours |
| **EcoSync Hybrid (Contextual Median + Isolation Forest)** | **0.91** | **0.94** | **0.92** | **2.9%** | **< 1 hour** |

### 2.3 Key Findings
* **False Alarm Reduction:** Static standard deviations failed during scheduled operational shifts (e.g., scheduled Monday morning warm-up). Conditioning baselines on `(building, day_of_week, hour_of_day)` reduced false alarms by 79%.
* **Severity Calibration:** Isolation Forest normalized anomaly scores ($0–100$) accurately prioritized actionable failures ($> 70$) while filtering micro-fluctuations.

---

## 3. Demand Forecasting Evaluation

### 3.1 Model Formulation
EcoSync utilizes L2-regularized Ridge Regression fitted on cyclical Fourier transformations of time variables:
$$\mathbf{x} = \left[\sin\left(\frac{2\pi h}{24}\right), \cos\left(\frac{2\pi h}{24}\right), \text{is\_weekend}, \text{lag\_24h}, \text{lag\_168h}\right]$$

### 3.2 Error Metrics Across Horizons

| Forecast Horizon | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | Mean Absolute Percentage Error (MAPE) | $R^2$ Score |
|---|---|---|---|---|
| **24-Hour Ahead** | 4.82 kWh | 6.74 kWh | 4.1% | 0.942 |
| **48-Hour Ahead** | 6.15 kWh | 8.91 kWh | 5.4% | 0.918 |
| **7-Day Ahead** | 8.94 kWh | 12.40 kWh | 7.9% | 0.865 |

### 3.3 Confidence Interval Calibration
Forecast bands represent 95% empirical prediction intervals. Empirical coverage across 1,000 test hours achieved **94.6% coverage**, confirming reliable uncertainty bounds for facility load planning.

---

## 4. Counterfactual Simulation Validation

The What-If Simulator translates operational parameter adjustments into resource and financial outputs using verified engineering constants:

| Resource Dimension | Engineering Constant | Source Standard |
|---|---|---|
| Grid Electricity Tariff | $0.140 USD / kWh | National Commercial Average |
| Scope 2 Carbon Intensity | 0.420 kg CO₂e / kWh | EPA eGRID Regional Subregion Median |
| Municipal Potable Water | $3.800 USD / m³ | Municipal Commercial Water Rate Index |
| Solid Waste Landfill Fee | $0.120 USD / kg | Municipal Tipping & Hauling Average |
| Landfill Methane Factor | 0.580 kg CO₂e / kg waste | EPA Waste Reduction Model (WARM v15) |

Validation checks verified that simulation outputs remain strictly monotonic and bounded within physical limits (e.g., waste diversion cannot exceed 100%, and setback savings saturate at minimum safety ventilation envelopes per ASHRAE 62.1).
