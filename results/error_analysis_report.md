# Detailed Model Error Analysis & Failure Mode Report

**Report Location**: [error_analysis_report.md](file:///home/dharani/Desktop/solar_flare/results/error_analysis_report.md)  
**Evaluation Scope**: Analysis of `41` False Negatives, `61` False Positives, and `71` Boundary Samples  

---

## 1. False Negative Analysis (Missed Major Flares: 41 Events)

False Negatives represent high-energy M/X major flares that the model failed to flag. Below are the top missed major flares ranked by lowest predicted probability:

| Index | Date | GOES Class | Predicted Prob | SNR | Net Peak Counts | Duration (s) | Failure Attribution |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 331 | `20260701` | `M8.5` | `0.0702` | `6.64` | `76.0` | `84s` | Low pre-flare SNR / gradual rise |
| 49 | `20240519` | `M1.6` | `0.1201` | `7.64` | `16.0` | `185s` | Low pre-flare SNR / gradual rise |
| 421 | `20260705` | `M1.3` | `0.1933` | `90.15` | `727.0` | `917s` | Short lookback window pulse |
| 291 | `20260630` | `M1.3` | `0.2032` | `15.92` | `244.0` | `4022s` | Short lookback window pulse |
| 88 | `20240523` | `M1.0` | `0.2200` | `7.80` | `104.0` | `380s` | Low pre-flare SNR / gradual rise |
| 524 | `20260720` | `M3.4` | `0.2640` | `11.35` | `47.0` | `405s` | Short lookback window pulse |
| 417 | `20260704` | `M1.1` | `0.2766` | `23.32` | `376.0` | `795s` | Short lookback window pulse |
| 194 | `20240530` | `M1.0` | `0.3273` | `5.95` | `84.0` | `79s` | Low pre-flare SNR / gradual rise |
| 175 | `20240529` | `M1.4` | `0.3276` | `37.82` | `552.1` | `1037s` | Short lookback window pulse |
| 418 | `20260704` | `M1.0` | `0.3276` | `23.44` | `288.0` | `1122s` | Short lookback window pulse |
| 173 | `20240529` | `M1.3` | `0.3298` | `41.76` | `561.0` | `912s` | Short lookback window pulse |
| 726 | `20260821` | `M1.6` | `0.3309` | `6.31` | `190.0` | `157s` | Low pre-flare SNR / gradual rise |
| 94 | `20240523` | `M2.5` | `0.3368` | `18.17` | `617.0` | `4739s` | Short lookback window pulse |
| 118 | `20240524` | `M1.4` | `0.3424` | `15.89` | `384.0` | `4139s` | Short lookback window pulse |
| 25 | `20240516` | `M1.0` | `0.3451` | `73.94` | `499.0` | `1180s` | Short lookback window pulse |

---

## 2. False Positive Analysis (Misclassified Minor Flares: 61 Events)

False Positives represent C-class or B-class flares that exhibited intense pre-flare acceleration mimicking major flare signatures:

| Index | Date | GOES Class | Predicted Prob | SNR | Net Peak Counts | Attribution |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 416 | `20260704` | `UNMATCHED` | `0.9455` | `13.19` | `323.1` | High pre-flare background flux | 
| 701 | `20260820` | `UNMATCHED` | `0.9374` | `9.68` | `110.0` | High pre-flare background flux | 
| 540 | `20260721` | `UNMATCHED` | `0.9229` | `10.15` | `136.0` | High pre-flare background flux | 
| 224 | `20260624` | `C3.5` | `0.9052` | `11.82` | `59.0` | High pre-flare background flux | 
| 352 | `20260703` | `C4.3` | `0.9045` | `18.76` | `124.0` | High pre-flare background flux | 
| 220 | `20240531` | `UNMATCHED` | `0.8977` | `6.73` | `69.0` | High pre-flare background flux | 
| 306 | `20260630` | `UNMATCHED` | `0.8969` | `9.57` | `194.0` | High pre-flare background flux | 
| 129 | `20240526` | `C2.4` | `0.8669` | `7.12` | `47.0` | High pre-flare background flux | 
| 307 | `20260630` | `UNMATCHED` | `0.8580` | `6.21` | `80.0` | High pre-flare background flux | 
| 450 | `20260705` | `C9.5` | `0.8556` | `18.19` | `247.0` | High pre-flare background flux | 
| 519 | `20260720` | `C7.1` | `0.8507` | `63.72` | `371.0` | High pre-flare background flux | 
| 317 | `20260701` | `C4.3` | `0.8396` | `29.31` | `156.0` | High pre-flare background flux | 
| 541 | `20260721` | `UNMATCHED` | `0.8307` | `12.21` | `135.0` | High pre-flare background flux | 
| 183 | `20240529` | `UNMATCHED` | `0.8155` | `14.97` | `518.1` | High pre-flare background flux | 
| 426 | `20260705` | `C7.7` | `0.7790` | `43.57` | `371.0` | High pre-flare background flux | 

---

## 3. Probability Distribution & Boundary Analysis

- **True Positive Mean Probability**: `0.7921`
- **True Negative Mean Probability**: `0.2663`
- **False Positive Mean Probability**: `0.6800`
- **False Negative Mean Probability**: `0.3659`
- **Hard Boundary Count [0.40, 0.60]**: `71` samples
