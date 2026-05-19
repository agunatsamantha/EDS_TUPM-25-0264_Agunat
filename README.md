EDS-TUPM-25-0264-AGUNAT
Engineering Data Systems Pipeline

Topic: Renewable Energy Systems Analytics - Power Curve Deviation Analysis
Course: Computer Programming 1
Academic Year: 2026

---

## Project Description

This project implements a Python-based data analytics pipeline for analyzing wind turbine performance using SCADA data. The system focuses on detecting power curve deviations between actual and theoretical turbine output.

Using the Wind Turbine SCADA Dataset from Kaggle, the pipeline performs:

- Automated data cleaning and preprocessing  
- Power curve deviation computation  
- Descriptive statistical analysis (mean, standard deviation, min, max)  
- Adaptive Multi-Layer Underperformance Filtering  
- High-Deviation Severity Detection using statistical methods  
- Static and animated visualizations for performance monitoring  

---

## Dataset

Wind Turbine SCADA Dataset (Kaggle)  
Includes wind speed, actual power output, theoretical power curve, and operational metrics.

---

## Analytical Methods

This project uses:

- Descriptive Statistics (NumPy-based)
- Comparative Analysis (Actual vs Theoretical Power)
- Z-score anomaly detection (SciPy-based)
- Layered performance classification system

---

## Custom Filters

### Adaptive Multi-Layer Underperformance Filter
Analyzes turbine performance across wind speed ranges to identify efficiency variations under different operating conditions.

### High-Deviation Severity Filter
Detects abnormal performance behavior using statistical deviation (Z-score) analysis.

---

## Visualizations

The system generates:

- Wind Speed vs Actual Power Curve  
- Actual vs Theoretical Power Comparison  
- Deviation Distribution Histogram  
- Animated Power Curve Evolution (GIF)  
- Animated Deviation Trend Analysis (GIF)  

---


