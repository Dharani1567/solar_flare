# Solar Flare Forecasting Skill Scores Report

**Report Location**: [solar_forecast_metrics_report.md](file:///home/dharani/Desktop/solar_flare/results/solar_forecast_metrics_report.md)  
**Evaluation Standard**: Solar Physics & Space Weather Forecasting Skill Metrics  
**Optimal Decision Threshold**: `0.35`  

---

## 1. Primary Operational Skill Scores (Threshold = 0.35)

| Skill Metric Symbol | Metric Full Name | Metric Value | Benchmark Interpretation |
| :--- | :--- | :---: | :--- |
| **TSS** | **True Skill Statistic** | **`0.6752`** | Net operational skill over random chance ($[-1, +1]$ scale) |
| **HSS** | **Heidke Skill Score** | **`0.4999`** | Skill relative to random reference forecast ($[- \infty, +1]$ scale) |
| **POD** | **Probability of Detection (Sensitivity)** | **`0.8596`** | Proportion of actual major flares correctly predicted |
| **FAR** | **False Alarm Ratio** | **`0.5377`** | Proportion of positive predictions that were false alarms |
| **CSI** | **Critical Success Index (Threat Score)** | **`0.4298`** | Ratio of TP to total positive forecast/event occurrences |

---

## 2. Mathematical Metric Formulations

$$ \text{TSS} = \text{POD} - \text{POFD} = \frac{\text{TP}}{\text{TP} + \text{FN}} - \frac{\text{FP}}{\text{FP} + \text{TN}} $$

$$ \text{HSS} = \frac{2 (\text{TP} \cdot \text{TN} - \text{FP} \cdot \text{FN})}{(\text{TP} + \text{FN})(\text{FN} + \text{TN}) + (\text{TP} + \text{FP})(\text{FP} + \text{TN})} $$

$$ \text{CSI} = \frac{\text{TP}}{\text{TP} + \text{FP} + \text{FN}} $$
