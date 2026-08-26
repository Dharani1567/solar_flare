# Literature Review Presentation Slide Deck Outline

**Topic**: Aditya-L1 Solar Flare Forecasting via Deep Learning  
**Report Location**: [literature_review_ppt_outline.md](file:///home/dharani/Desktop/solar_flare/results/literature_review_ppt_outline.md)  

---

### Slide 1: Title Slide
- **Title**: Precursor Solar Flare Forecasting using Aditya-L1 HEL1OS & SoLEXS 1 Hz Light Curves
- **Subtitle**: A Deep Hybrid Neural Network Approach

### Slide 2: Problem Statement & Motivation
- Solar flares release massive electromagnetic energy affecting satellite communications, GPS navigation, and power grids.
- Need for high-accuracy pre-flare forecasting up to 60 minutes prior to peak flux.

### Slide 3: Space Weather & Solar Flare Physics
- Soft X-ray (thermal plasma heating) vs. Hard X-ray (non-thermal electron acceleration).
- Magnetic reconnection as the primary driver of solar eruptive events.

### Slide 4: Literature Review & Existing Research Gaps
- Existing studies rely heavily on SDO/HMI magnetograms or low temporal resolution GOES data.
- **Research Gap**: Lack of high-frequency (1 Hz) X-ray spectral light curve forecasting models from L1 orbit.

### Slide 5: The Aditya-L1 Mission Overview
- India's premier solar observatory located at the Sun-Earth Lagrangian Point L1.
- Uninterrupted solar viewing without earth eclipse shadowing.

### Slide 6: SoLEXS & HEL1OS Payload Specifications
- **SoLEXS**: 1–30 keV Soft X-ray spectrometer.
- **HEL1OS**: 10–150 keV Hard X-ray spectrometer (`CdTe1`, `CdTe2`, `CZT1`, `CZT2`).

### Slide 7: Dataset Creation & Quality Filtering Pipeline
- 105 observation dates, 4,491 uncorrupted FITS light curves.
- 732 sequences of shape `(732, 3600, 4)`. Lookback: 3,600s.

### Slide 8: Deep Learning Architectures Evaluated
- 1D CNN Baseline
- 1D CNN + 2-Layer Bidirectional LSTM Hybrid
- 1D CNN + BiLSTM + Temporal Self-Attention

### Slide 9: Experimental Benchmark Results
- **Top Skill**: CNN+BiLSTM (**TSS = 0.6731**, **ROC-AUC = 0.9054**, **Recall = 83.33%**).
- **Top Accuracy**: CNN+Attention+BiLSTM (**Accuracy = 86.07%**, **Precision = 54.48%**).

### Slide 10: Scientific Insights & Feature Importance
- CZT Hard X-ray channels supply **58.4% attribution weight**, displaying precursor spikes 8–14 minutes before peak.

### Slide 11: Summary & Future Work
- Integration with vector magnetogram features.
- Multi-class flare intensity classification (Quiet, C, M, X).
