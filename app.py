"""
Aditya-L1 Solar Flare Forecasting System — College Project Demo Application
Streamlit interactive dashboard for real-time solar flare early warning
using ISRO Aditya-L1 HEL1OS and SoLEXS X-ray spectrometer telemetry.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import streamlit as st
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from models import (
    CNNBiLSTMModel,
    CNNAttentionBiLSTMModel,
    TCNModel,
    InceptionTimeModel,
    ResNet1DModel
)

st.set_page_config(
    page_title="Aditya-L1 Solar Flare Forecasting AI",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

V2_DIR = PROJECT_ROOT / "dataset_forecast_v2"
MODELS_DIR = PROJECT_ROOT / "results" / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
SAMPLE_DIR = PROJECT_ROOT / "sample_test_inputs"

# Model configurations
MODEL_REGISTRY = {
    "TCN (Temporal Convolutional Network) — [Rank 1, Best Model]": {
        "class": TCNModel,
        "kwargs": {"in_channels": 4, "num_classes": 2},
        "ckpt_name": "final_tcn.pt",
        "fallback_ckpt": "tcn_fold_1.pt",
        "optimal_th": 0.73,
        "roc_auc": 0.9293,
        "f1": 0.7632,
        "recall": 0.9165,
        "type": "Dilated Temporal Convolutions"
    },
    "ResNet1D (Deep Residual Network) — [Rank 2]": {
        "class": ResNet1DModel,
        "kwargs": {"in_channels": 4, "num_classes": 2},
        "ckpt_name": "final_resnet1d.pt",
        "fallback_ckpt": "resnet1d_fold_1.pt",
        "optimal_th": 0.15,
        "roc_auc": 0.9078,
        "f1": 0.7477,
        "recall": 1.0000,
        "type": "Residual Convolutions"
    },
    "CNN + BiLSTM (Hybrid Temporal Model) — [Rank 3]": {
        "class": CNNBiLSTMModel,
        "kwargs": {"in_channels": 4, "hidden_dim": 64, "num_classes": 2},
        "ckpt_name": "final_cnn_bilstm.pt",
        "fallback_ckpt": "cnn_bilstm_fold_1.pt",
        "optimal_th": 0.15,
        "roc_auc": 0.8800,
        "f1": 0.7477,
        "recall": 1.0000,
        "type": "CNN + Recurrent BiLSTM"
    },
    "CNN + Attention + BiLSTM (Enhanced Hybrid) — [Rank 4]": {
        "class": CNNAttentionBiLSTMModel,
        "kwargs": {"in_channels": 4, "hidden_dim": 64, "num_classes": 2},
        "ckpt_name": "final_cnn_attention_bilstm.pt",
        "fallback_ckpt": "cnn_attention_bilstm_fold_1.pt",
        "optimal_th": 0.15,
        "roc_auc": 0.8778,
        "f1": 0.7477,
        "recall": 1.0000,
        "type": "Temporal Self-Attention + BiLSTM"
    },
    "InceptionTime (Multi-Scale Inception) — [Rank 5]": {
        "class": InceptionTimeModel,
        "kwargs": {"in_channels": 4, "num_classes": 2},
        "ckpt_name": "final_inceptiontime.pt",
        "fallback_ckpt": "inceptiontime_fold_1.pt",
        "optimal_th": 0.17,
        "roc_auc": 0.8703,
        "f1": 0.7477,
        "recall": 1.0000,
        "type": "Multi-Scale 1D Inception"
    }
}


@st.cache_resource
def load_deep_learning_model(model_name: str):
    cfg = MODEL_REGISTRY[model_name]
    model = cfg["class"](**cfg["kwargs"])
    ckpt_path = MODELS_DIR / cfg["ckpt_name"]
    if not ckpt_path.exists():
        ckpt_path = MODELS_DIR / cfg["fallback_ckpt"]

    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state, strict=False)
    model.eval()
    return model


@st.cache_data
def load_benchmark_csv():
    csv_p = RESULTS_DIR / "benchmark_comparison.csv"
    if not csv_p.exists():
        csv_p = PROJECT_ROOT / "benchmark_comparison.csv"
    if csv_p.exists():
        return pd.read_csv(csv_p)
    return None


@st.cache_data
def load_ml_csv():
    csv_p = RESULTS_DIR / "ml_comparison.csv"
    if not csv_p.exists():
        csv_p = PROJECT_ROOT / "ml_comparison.csv"
    if csv_p.exists():
        return pd.read_csv(csv_p)
    return None


@st.cache_data
def load_ml_vs_dl_csv():
    csv_p = RESULTS_DIR / "ml_vs_dl_comparison.csv"
    if not csv_p.exists():
        csv_p = PROJECT_ROOT / "ml_vs_dl_comparison.csv"
    if csv_p.exists():
        return pd.read_csv(csv_p)
    return None


def run_model_inference(model, seq: np.ndarray) -> float:
    # seq shape: (3600, 4)
    # Standardize input
    mean = np.mean(seq, axis=0, keepdims=True)
    std = np.std(seq, axis=0, keepdims=True) + 1e-6
    norm_seq = ((seq - mean) / std).astype(np.float32)

    tensor_in = torch.from_numpy(norm_seq).unsqueeze(0)  # (1, 3600, 4)
    with torch.no_grad():
        logits = model(tensor_in)
        prob_major = torch.softmax(logits, dim=1)[0, 1].item()
    return float(prob_major)


def plot_telemetry_channels(seq: np.ndarray, title: str):
    time_min = np.linspace(0, 60, len(seq))
    fig, axes = plt.subplots(4, 1, figsize=(11, 7), sharex=True)

    channels = [
        ("HEL1OS: CdTe1 (10–25 keV Soft X-ray)", seq[:, 0], "#1D3557"),
        ("HEL1OS: CdTe2 (25–60 keV Med X-ray)", seq[:, 1], "#457B9D"),
        ("HEL1OS: CZT1 (60–150 keV Hard X-ray)", seq[:, 2], "#E63946"),
        ("SoLEXS: Solar X-ray (1–30 keV Total Flux)", seq[:, 3], "#E76F51")
    ]

    for ax, (label, data, color) in zip(axes, channels):
        ax.plot(time_min, data, color=color, linewidth=1.2, label=label)
        ax.set_ylabel("Count Rate (s⁻¹)", fontsize=9, fontweight="bold")
        ax.legend(loc="upper left", fontsize=8.5, frameon=True)
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[-1].set_xlabel("Observation Window Time (Minutes elapsed across 1 hour)", fontsize=10, fontweight="bold")
    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.99)
    plt.tight_layout()
    return fig


def main():
    # Header Banner
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1D3557 0%, #457B9D 100%); padding: 22px; border-radius: 12px; color: white; margin-bottom: 20px;">
        <h1 style="margin:0; font-size: 30px;">☀️ Aditya-L1 Solar Flare Forecasting AI Observatory</h1>
        <p style="margin: 6px 0 0 0; font-size: 15px; opacity: 0.92;">
            Early Warning System for Major (M/X Class) Solar Flares using ISRO Aditya-L1 HEL1OS & SoLEXS Spectrometer Telemetry
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/d/d7/Aditya_L1_Spacecraft_model.png/640px-Aditya_L1_Spacecraft_model.png", use_container_width=True)
    st.sidebar.title("🛰️ Model & Sensor Controls")

    selected_model_name = st.sidebar.selectbox(
        "Select Deep Learning Architecture:",
        list(MODEL_REGISTRY.keys()),
        index=0
    )
    model_cfg = MODEL_REGISTRY[selected_model_name]
    model = load_deep_learning_model(selected_model_name)

    st.sidebar.markdown(f"""
    **Architecture Specifications:**
    - **Family**: `{model_cfg['type']}`
    - **ROC-AUC**: `{model_cfg['roc_auc']:.4f}`
    - **F1 Score**: `{model_cfg['f1']:.4f}`
    - **Recommended Threshold**: `{model_cfg['optimal_th']:.2f}`
    """)

    threshold = st.sidebar.slider(
        "Decision Threshold:",
        min_value=0.05,
        max_value=0.95,
        value=float(model_cfg["optimal_th"]),
        step=0.02,
        help="Probabilities above this threshold trigger an imminent Major Flare Alert."
    )

    st.sidebar.markdown("---")
    st.sidebar.info("""
    **College Project Details:**
    - **Mission**: ISRO Aditya-L1 (Sun-Earth L1 Lagrange Point)
    - **Payloads**: HEL1OS (Hard X-ray) & SoLEXS (Soft X-ray)
    - **Cadence**: 1-Hour continuous sequences (3,600s, 4 Channels)
    - **Target**: Forecasting Major M & X Class Flares before peak
    """)

    # Main Tabs
    tab_forecast, tab_benchmark = st.tabs([
        "🔮 Real-Time Flare Forecasting Demo",
        "📊 5-Model Benchmark Suite"
    ])

    # =================================================================
    # TAB 1: FORECASTING DEMO & UPLOAD
    # =================================================================
    with tab_forecast:
        st.subheader("1. Select or Upload Lightcurve Telemetry")

        input_mode = st.radio(
            "Choose Telemetry Input Method:",
            [
                "📁 Select Pre-loaded Scientific Event Sample",
                "📤 Upload Your Own Telemetry File (.csv or .npy)"
            ],
            horizontal=True
        )

        chosen_seq = None
        sample_description = ""
        ground_truth_label = None

        if "Pre-loaded" in input_mode:
            sample_choice = st.selectbox(
                "Choose Test Observation Scenario:",
                [
                    "Sample 1: Imminent M-Class Solar Flare Precursor (y=1, Major Flare)",
                    "Sample 2: Imminent Extreme X-Class Solar Flare Precursor (y=1, Extreme Event)",
                    "Sample 3: Quiescent Solar Background (y=0, Quiet Sun)"
                ]
            )

            if "M-Class" in sample_choice:
                seq_file = SAMPLE_DIR / "sample_1_major_flare_mclass.npy"
                ground_truth_label = 1
                sample_description = "Active Region Coronal Loop Heating preceding an M3.4 GOES Flare (Horizon: 3 Hours)"
            elif "X-Class" in sample_choice:
                seq_file = SAMPLE_DIR / "sample_2_major_flare_xclass.npy"
                ground_truth_label = 1
                sample_description = "Intense Magnetic Flux Buildup & Micro-bursts preceding an X2.1 Super-Flare (Horizon: 1 Hour)"
            else:
                seq_file = SAMPLE_DIR / "sample_3_quiet_solar_corona.npy"
                ground_truth_label = 0
                sample_description = "Quiescent Solar Corona with Baseline Orbit Thermal Drift (No Flare Precursor)"

            if seq_file.exists():
                chosen_seq = np.load(seq_file)

        else:
            st.markdown("#### Upload Custom Telemetry File")
            st.write("Upload a 1-hour time series of shape `(3600, 4)` as a `.csv` or `.npy` file.")

            col_up, col_down = st.columns([2, 1])

            with col_up:
                uploaded_file = st.file_uploader(
                    "Drop file here or click to browse:",
                    type=["csv", "npy"]
                )

            with col_down:
                st.markdown("**Need test files to test uploading?**")
                m_csv_path = SAMPLE_DIR / "sample_1_major_flare_mclass.csv"
                q_csv_path = SAMPLE_DIR / "sample_3_quiet_solar_corona.csv"
                if m_csv_path.exists():
                    with open(m_csv_path, "rb") as f:
                        st.download_button(
                            "📥 Download M-Class Flare Sample CSV",
                            f,
                            file_name="sample_mclass_flare.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                if q_csv_path.exists():
                    with open(q_csv_path, "rb") as f:
                        st.download_button(
                            "📥 Download Quiet Sun Sample CSV",
                            f,
                            file_name="sample_quiet_sun.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith(".npy"):
                        chosen_seq = np.load(uploaded_file)
                    else:
                        df = pd.read_csv(uploaded_file)
                        chosen_seq = df.values.astype(np.float32)

                    if chosen_seq.ndim == 3:
                        chosen_seq = chosen_seq[0]

                    if chosen_seq.shape != (3600, 4):
                        st.error(f"Shape Mismatch: Expected (3600, 4), got {chosen_seq.shape}.")
                        chosen_seq = None
                    else:
                        st.success(f"Successfully loaded `{uploaded_file.name}` with shape (3600, 4)!")
                        sample_description = f"User Uploaded File: {uploaded_file.name}"
                except Exception as e:
                    st.error(f"Error reading file: {e}")

        # Visualization & Prediction
        if chosen_seq is not None:
            st.markdown("---")
            st.subheader("2. Multi-Channel X-Ray Lightcurve Visualization")
            st.caption(sample_description)

            fig = plot_telemetry_channels(chosen_seq, f"Aditya-L1 Telemetry — {sample_description}")
            st.pyplot(fig)

            st.markdown("---")
            st.subheader("3. Model Forecasting Result")

            col_btn, _ = st.columns([1, 2])
            with col_btn:
                run_pred = st.button("🔮 Run Solar Flare Forecasting", type="primary", use_container_width=True)

            if run_pred:
                prob = run_model_inference(model, chosen_seq)
                is_flare = prob >= threshold

                col_res1, col_res2, col_res3 = st.columns(3)

                with col_res1:
                    st.metric("Imminent Flare Probability", f"{prob*100:.1f}%")
                with col_res2:
                    st.metric("Decision Threshold", f"{threshold:.2f}")
                with col_res3:
                    if ground_truth_label is not None:
                        gt_text = "Major Flare (y=1)" if ground_truth_label == 1 else "Quiet Sun (y=0)"
                        st.metric("Ground Truth State", gt_text)
                    else:
                        st.metric("Model Architecture", model_cfg["type"].split()[0])

                st.progress(float(np.clip(prob, 0.0, 1.0)))

                if is_flare:
                    st.error(f"""
                    ### 🚨 CRITICAL ALERT: MAJOR SOLAR FLARE IMMINENT (M/X CLASS)
                    - **Forecasted State**: **Major Solar Flare Imminent** ($y=1$)
                    - **Confidence Level**: **{prob*100:.1f}%** (exceeds operational threshold `{threshold:.2f}`)
                    - **Recommended Action**: Initiate satellite star-tracker shielding, notify ISRO ISTRAC telemetry control, and alert national power grid monitors.
                    """)
                else:
                    st.success(f"""
                    ### 🟢 STATUS ALL CLEAR: QUIESCENT SOLAR ACTIVITY
                    - **Forecasted State**: **Quiet Sun / Minor Micro-Flare Only** ($y=0$)
                    - **Confidence Level**: **{(1-prob)*100:.1f}%** safe margin below threshold `{threshold:.2f}`
                    - **Status**: Solar background emissions within safe nominal operating parameters.
                    """)

    # =================================================================
    # TAB 2: BENCHMARK SUITE (DL & ML)
    # =================================================================
    with tab_benchmark:
        st.subheader("🏆 Model Benchmark Suite (Deep Learning vs Classical ML)")
        st.caption("All models evaluated using Stratified Group 5-Fold Cross-Validation on 2,300 Aditya-L1 observations (strictly grouped by flare_event_id for 0.00% leakage).")

        # Head to Head comparison
        st.markdown("### 🥊 Head-to-Head: Deep Learning vs Classical Machine Learning")
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Top DL Model (TCN)", "0.9293 ROC-AUC", "+7.7% vs Top ML")
        col_m2.metric("Top ML Model (KNN)", "0.8620 ROC-AUC", "Baseline Tabular")
        col_m3.metric("DL F1 Advantage", "0.7632 vs 0.6831", "+11.7% F1 Score")

        df_ml_dl = load_ml_vs_dl_csv()
        if df_ml_dl is not None:
            st.dataframe(df_ml_dl, use_container_width=True, hide_index=True)

        ml_vs_dl_path = PROJECT_ROOT / "ml_vs_dl_comparison.png"
        if ml_vs_dl_path.exists():
            st.image(str(ml_vs_dl_path), caption="ROC-AUC & F1 Score Comparison: Temporal Deep Learning Architectures vs Classical Machine Learning Baselines", use_container_width=True)

        st.markdown("---")
        st.subheader("🧠 Deep Learning Architectures (5 Models)")
        df_bm = load_benchmark_csv()
        if df_bm is not None:
            df_filtered = df_bm[df_bm["Model Architecture"] != "1D CNN"].copy()
            df_filtered["Rank"] = range(1, len(df_filtered) + 1)
            st.dataframe(df_filtered, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("🌲 Classical Machine Learning Baselines (5 Models)")
        df_ml = load_ml_csv()
        if df_ml is not None:
            st.dataframe(df_ml, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("📈 Deep Learning Detailed Diagnostic Charts")

        col_p1, col_p2 = st.columns(2)

        with col_p1:
            roc_path = PROJECT_ROOT / "roc_curve_comparison.png"
            if roc_path.exists():
                st.image(str(roc_path), caption="ROC Curves: TCN leads with AUC = 0.9293", use_container_width=True)

            cm_path = PROJECT_ROOT / "confusion_matrix_best_model.png"
            if cm_path.exists():
                st.image(str(cm_path), caption="Confusion Matrix — Top Model: TCN", use_container_width=True)

        with col_p2:
            pr_path = PROJECT_ROOT / "precision_recall_comparison.png"
            if pr_path.exists():
                st.image(str(pr_path), caption="Precision-Recall Curves against 25% Baseline Prevalence", use_container_width=True)

            loss_path = PROJECT_ROOT / "training_curves.png"
            if loss_path.exists():
                st.image(str(loss_path), caption="5-Fold Cross-Entropy Validation Loss Trajectories", use_container_width=True)

        st.markdown("---")
        st.subheader("🏗️ Neural Network Architecture Visualizations (Individual & Combined)")
        st.caption("Inspect holographic 3D AI architecture diagrams for each deep learning model.")

        arch_view_mode = st.radio(
            "Select Architecture View:",
            [
                "🔬 Individual Model Deep Dive (Single Separate Photos)",
                "🌌 Combined Multi-Model 3D AI Overview"
            ],
            horizontal=True
        )

        if "Individual" in arch_view_mode:
            arch_choice = st.selectbox(
                "Choose Architecture to Inspect:",
                [
                    "Rank 1: TCN (Temporal Convolutional Network)",
                    "Rank 2: ResNet1D (Deep 1D Residual Network)",
                    "Rank 3: CNN + BiLSTM (Convolutional Recurrent Hybrid)",
                    "Rank 4: CNN + Attention + BiLSTM (Attention-Gated Recurrent)",
                    "Rank 5: InceptionTime (Multi-Scale Inception Network)"
                ]
            )

            arch_file_map = {
                "Rank 1: TCN (Temporal Convolutional Network)": ("tcn_architecture.png", "Temporal Convolutional Network: Dilated causal 1D convolutions with receptive field expansion (d=1, 2, 4) & residual bypass."),
                "Rank 2: ResNet1D (Deep 1D Residual Network)": ("resnet1d_architecture.png", "ResNet1D: Multi-stage residual blocks with identity shortcut connections and global average pooling."),
                "Rank 3: CNN + BiLSTM (Convolutional Recurrent Hybrid)": ("cnn_bilstm_architecture.png", "CNN + BiLSTM: Dual-stage Conv1D feature extraction feeding dual-layer Bidirectional LSTM recurrent units."),
                "Rank 4: CNN + Attention + BiLSTM (Attention-Gated Recurrent)": ("cnn_attention_bilstm_architecture.png", "CNN + Attention + BiLSTM: Self-attention mechanism dynamically weighting temporal precursor feature vectors."),
                "Rank 5: InceptionTime (Multi-Scale Inception Network)": ("inceptiontime_architecture.png", "InceptionTime: Parallel multi-kernel inception filters (k=10, 20, 40) with 1x1 bottleneck layer.")
            }

            img_fname, img_caption = arch_file_map[arch_choice]
            single_img_path = PROJECT_ROOT / img_fname
            if single_img_path.exists():
                st.image(str(single_img_path), caption=img_caption, use_container_width=True)
        else:
            arch_path = PROJECT_ROOT / "all_models_architecture.png"
            if arch_path.exists():
                st.image(str(arch_path), caption="End-to-End Holographic 3D AI Architecture Suite for Solar Flare Forecasting", use_container_width=True)

        tbl_path = PROJECT_ROOT / "model_ranking_table.png"
        if tbl_path.exists():
            st.image(str(tbl_path), caption="Phase-2 Deep Learning Benchmark Ranking Table", use_container_width=True)


if __name__ == "__main__":
    main()
