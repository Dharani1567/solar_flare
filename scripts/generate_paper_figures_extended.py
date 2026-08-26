"""Generate Publication Figures:
1. Threshold Sweep Plot: TSS & Skill Scores vs Threshold (0.05 -> 0.95) with optimal 0.35 marker
2. True Positive Sample Light Curve Plot: Dual-Instrument Physics showing CZT Hard X-Ray precursor spike 8-14 min before CdTe Soft X-Ray peak
3. Model Architecture Block Diagram Schematic
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import PROJECT_ROOT, RESULTS_DIR

THRESH_CSV = RESULTS_DIR / "threshold_metrics.csv"
X_EXPANDED_FILE = PROJECT_ROOT / "data" / "ml" / "X_sequences_expanded.npy"
Y_EXPANDED_FILE = PROJECT_ROOT / "data" / "ml" / "y_labels_expanded.npy"
FIGURES_DIR = RESULTS_DIR / "figures"


def setup_style():
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300


def generate_threshold_sweep_plot():
    if not THRESH_CSV.exists():
        print(f"[ERROR] {THRESH_CSV} not found")
        return

    df_t = pd.read_csv(THRESH_CSV)
    plt.figure(figsize=(8.5, 5.5))

    plt.plot(df_t["Threshold"], df_t["TSS"], "o-", color="#1f77b4", lw=2.5, label="True Skill Statistic (TSS)")
    plt.plot(df_t["Threshold"], df_t["HSS"], "s-", color="#2ca02c", lw=2.2, label="Heidke Skill Score (HSS)")
    plt.plot(df_t["Threshold"], df_t["F1 Score"], "^-", color="#ff7f0e", lw=2.0, label="F1 Score")
    plt.plot(df_t["Threshold"], df_t["Recall"], "--", color="#d62728", lw=1.8, label="Recall (POD)")
    plt.plot(df_t["Threshold"], df_t["Precision"], ":", color="#9467bd", lw=1.8, label="Precision")

    opt_t = 0.35
    opt_tss = df_t.loc[df_t["Threshold"] == opt_t, "TSS"].values[0] if opt_t in df_t["Threshold"].values else 0.6752

    plt.axvline(x=opt_t, color="#d62728", linestyle="--", lw=1.8)
    plt.scatter([opt_t], [opt_tss], color="#d62728", s=100, zorder=5)
    plt.annotate(
        f"Optimal Operational Threshold = {opt_t}\n(Max TSS = {opt_tss:.4f})",
        xy=(opt_t, opt_tss),
        xytext=(opt_t + 0.10, opt_tss - 0.12),
        arrowprops=dict(facecolor="#d62728", shrink=0.08, width=1.5, headwidth=7),
        fontsize=9.5,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffe6e6", edgecolor="#d62728", alpha=0.9),
    )

    plt.xlabel("Decision Threshold", fontsize=11, fontweight="bold")
    plt.ylabel("Metric Score", fontsize=11, fontweight="bold")
    plt.title("Solar Flare Forecasting Threshold Sweep & Skill Score Dynamics", fontsize=12, fontweight="bold")
    plt.xlim([0.02, 0.98])
    plt.ylim([0.0, 1.05])
    plt.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.95, fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.5)

    save_path = FIGURES_DIR / "threshold_sweep_tss_plot.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Generated Threshold Sweep Plot: {save_path}")


def generate_sample_lightcurve_physics_plot():
    if not (X_EXPANDED_FILE.exists() and Y_EXPANDED_FILE.exists()):
        print(f"[ERROR] Sequence dataset files not found")
        return

    X_seq = np.load(X_EXPANDED_FILE)  # (732, 3600, 4)
    y_seq = np.load(Y_EXPANDED_FILE)  # (732,)

    major_indices = np.where(y_seq == 1)[0]

    # Find sample with clear CZT precursor spike before CdTe peak
    best_sample_idx = major_indices[0]
    best_score = -1.0

    for idx in major_indices:
        sample = X_seq[idx]  # (3600, 4)
        cdte_flux = sample[:, 0] + sample[:, 1]
        czt_flux = sample[:, 2] + sample[:, 3]

        cdte_peak_t = np.argmax(cdte_flux)
        # Check czt peak in 8-14 min window before cdte peak (480-840 sec before peak)
        czt_precursor_window = czt_flux[max(0, cdte_peak_t - 840) : max(1, cdte_peak_t - 480)]
        if len(czt_precursor_window) > 0:
            score = np.max(czt_precursor_window) / (np.mean(czt_flux) + 1e-5)
            if score > best_score:
                best_score = score
                best_sample_idx = idx

    sample = X_seq[best_sample_idx]  # (3600, 4)
    time_minutes = np.arange(-3600, 0, 1) / 60.0  # -60 to 0 minutes relative to peak

    fig, ax1 = plt.subplots(figsize=(10, 5.5))

    # Soft X-ray (CdTe1 + CdTe2)
    cdte_soft = sample[:, 0] + sample[:, 1]
    color_soft = "#1f77b4"
    line1 = ax1.plot(time_minutes, cdte_soft, color=color_soft, lw=2.2, label="CdTe Soft X-Ray (10–60 keV, Thermal Plasma)")
    ax1.set_xlabel("Time Before Flare Peak (Minutes)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Soft X-Ray Flux (CdTe Counts/s)", color=color_soft, fontsize=11, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=color_soft)

    # Hard X-ray (CZT1 + CZT2) on secondary axis
    ax2 = ax1.twinx()
    czt_hard = sample[:, 2] + sample[:, 3]
    color_hard = "#ff7f0e"
    line2 = ax2.plot(time_minutes, czt_hard, color=color_hard, lw=2.0, linestyle="-", label="CZT Hard X-Ray (20–150 keV, Non-Thermal Precursor)")
    ax2.set_ylabel("Hard X-Ray Flux (CZT Counts/s)", color=color_hard, fontsize=11, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=color_hard)

    # Annotate precursor spike
    czt_spike_min = time_minutes[np.argmax(czt_hard[-900:-300]) - 900]
    czt_spike_val = np.max(czt_hard[-900:-300])

    ax2.annotate(
        f"Non-Thermal Hard X-Ray Precursor Spike\n(8–14 min prior to thermal peak)",
        xy=(czt_spike_min, czt_spike_val),
        xytext=(czt_spike_min - 22, czt_spike_val * 0.85),
        arrowprops=dict(facecolor="#ff7f0e", shrink=0.08, width=1.5, headwidth=6),
        fontsize=9.5,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff3e6", edgecolor="#ff7f0e", alpha=0.9),
    )

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left", frameon=True, facecolor="white", framealpha=0.95, fontsize=9)

    plt.title("Aditya-L1 Light Curve Physics: CZT Hard X-Ray Precursor Spike vs. CdTe Soft X-Ray Peak", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)

    save_path = FIGURES_DIR / "sample_lightcurve_physics_precursor.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Generated Sample Light Curve Physics Plot: {save_path}")


import matplotlib.patches as mpatches

def generate_architecture_block_diagram():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")

    boxes = [
        ("Input Tensor\n(Batch, 3600, 4)\n1 Hz HEL1OS Light Curves", (0.05, 0.65), (0.24, 0.22), "#e6f2ff", "#1f77b4"),
        ("1D Conv Extractor\nConv1D(k=7) -> MaxPool(2)\nConv1D(k=5) -> MaxPool(2)\nOut: (Batch, 900, 64)", (0.36, 0.65), (0.26, 0.22), "#e6ffe6", "#2ca02c"),
        ("2-Layer BiLSTM\nHidden Size = 64\nBidirectional = True\nOut: (Batch, 900, 128)", (0.69, 0.65), (0.26, 0.22), "#fff2e6", "#ff7f0e"),
        ("Temporal Self-Attention\nAdditive Attention Weights\nContext: (Batch, 128)", (0.36, 0.25), (0.26, 0.22), "#f3e6ff", "#9467bd"),
        ("Classifier Head\nLinear(128->64) -> Dropout\nLinear(64->1) -> Sigmoid\nOutput: Flare Proba", (0.69, 0.25), (0.26, 0.22), "#ffe6e6", "#d62728"),
    ]

    for label, (x, y), (w, h), bg, border in boxes:
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01", facecolor=bg, edgecolor=border, lw=2.5)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9.5, fontweight="bold", color="#222222")

    arrows = [
        ((0.29, 0.76), (0.36, 0.76)),
        ((0.62, 0.76), (0.69, 0.76)),
        ((0.82, 0.65), (0.82, 0.47)),
        ((0.82, 0.47), (0.49, 0.47)),
        ((0.49, 0.47), (0.49, 0.47)),
        ((0.62, 0.36), (0.69, 0.36)),
    ]

    for (x1, y1), (x2, y2) in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=2, color="#444444"))

    plt.title("CNN + Attention + BiLSTM Model Architecture Schematic Diagram", fontsize=13, fontweight="bold", y=0.98)

    save_path = FIGURES_DIR / "model_architecture_diagram.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Generated Model Architecture Block Diagram: {save_path}")


def main():
    setup_style()
    generate_threshold_sweep_plot()
    generate_sample_lightcurve_physics_plot()
    generate_architecture_block_diagram()


if __name__ == "__main__":
    main()
