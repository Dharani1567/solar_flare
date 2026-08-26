# Reproducibility Gap & Dependency Audit Report

**Report Location**: `results/reproducibility_gaps.md`  

---

## 1. Audited Reproducibility Considerations

1. **ISSDC PRADAN Portal Authentication**:
   - **Gap**: Level-1 observation archives are hosted on the ISSDC PRADAN portal (`https://pradan.issdc.gov.in`), requiring active Keycloak SSO login.
   - **Resolution**: Detailed access instructions and date range specifications are documented in `DATA.md`.

2. **Satellite Orbital Gaps**:
   - **Gap**: HEL1OS satellite orbital passages create intermittent data gaps.
   - **Resolution**: Quality threshold $N \ge 2,500$ samples per 3,600s window ensures only continuous, uncorrupted light curves are included in tensor `X_sequences_expanded.npy`.

3. **CPU vs. GPU Compute Execution Times**:
   - **Gap**: PyTorch model training runs on CPU (8 threads).
   - **Resolution**: Exact hyperparameters (`epochs=35, patience=5, lr=2e-3`) are configured for deterministic cross-validation reproducibility across CPU and CUDA backends.
