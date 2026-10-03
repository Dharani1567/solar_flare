"""
Generate High-Resolution Publication-Quality Architecture Diagram
for all 5 Deep Learning Models in the Aditya-L1 Solar Flare Forecasting Suite.
Excludes 1D CNN per project specification.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def draw_rounded_box(ax, x, y, w, h, text, bg_color, border_color="#1D3557", fontsize=8.5, fontweight="bold", text_color="#1D3557", radius=0.015):
    box = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.01,rounding_size={radius}",
        facecolor=bg_color,
        edgecolor=border_color,
        linewidth=1.3,
        zorder=3
    )
    ax.add_patch(box)
    lines = text.split("\n")
    if len(lines) == 1:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, fontweight=fontweight, color=text_color, zorder=4)
    else:
        # Multi-line
        total_h = h
        line_spacing = total_h / (len(lines) + 1)
        for i, line in enumerate(lines):
            weight = fontweight if i == 0 else "normal"
            size = fontsize if i == 0 else fontsize - 0.8
            ax.text(x + w / 2, y + h - (i + 1) * line_spacing, line, ha="center", va="center", fontsize=size, fontweight=weight, color=text_color, zorder=4)

def draw_arrow(ax, x1, y1, x2, y2, color="#457B9D", lw=1.5):
    ax.annotate(
        "",
        xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=12, shrinkA=2, shrinkB=2),
        zorder=2
    )

def generate_architectures_figure(output_path: str):
    fig, ax = plt.subplots(figsize=(24, 14), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Background canvas
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    # Title Banner
    banner = patches.FancyBboxPatch(
        (1.5, 93), 97, 6.2,
        boxstyle="round,pad=0.01,rounding_size=0.015",
        facecolor="#1D3557",
        edgecolor="#0D1B2A",
        linewidth=1.8,
        zorder=3
    )
    ax.add_patch(banner)
    ax.text(50, 97.2, "ISRO Aditya-L1 Solar Flare Forecasting — Deep Learning Architecture Suite", ha="center", va="center", fontsize=19, fontweight="bold", color="#FFFFFF", zorder=4)
    ax.text(50, 94.6, "End-to-End Temporal Models for 1-Hour Multi-Channel X-Ray Lightcurves (CdTe1, CdTe2, CZT1, SoLEXS) — Input Shape: (Batch, 4 Channels, 3600 Timesteps)", ha="center", va="center", fontsize=11, color="#E0E1DD", zorder=4)

    # 5 Model Columns
    col_width = 18.2
    col_gap = 1.3
    start_x = 2.0

    models_meta = [
        {
            "name": "TCN (Rank 1)",
            "subtitle": "Temporal Convolutional Network",
            "metric": "ROC-AUC: 0.9293 | F1: 0.7632",
            "tag_color": "#2A9D8F",
            "header_bg": "#E8F8F5",
            "blocks": [
                ("Input Sequence\n(Batch, 4, 3600)", "#E0E1DD", "#333333"),
                ("TCN Block 1 (d=1)\nConv1d(4→32, k=3) + BN + ReLU\nConv1d(32→32, k=3) + BN\n+ Residual Shortcut (1x1)", "#D8E2DC", "#1D3557"),
                ("MaxPool1d(stride=2)\nShape: (Batch, 32, 1800)", "#FFE5D9", "#555555"),
                ("TCN Block 2 (d=2)\nConv1d(32→64, k=3) + BN + ReLU\nConv1d(64→64, k=3) + BN\n+ Residual Shortcut (1x1)", "#D8E2DC", "#1D3557"),
                ("MaxPool1d(stride=2)\nShape: (Batch, 64, 900)", "#FFE5D9", "#555555"),
                ("TCN Block 3 (d=4)\nConv1d(64→128, k=3) + BN + ReLU\nConv1d(128→128, k=3) + BN\n+ Residual Shortcut (1x1)", "#D8E2DC", "#1D3557"),
                ("AdaptiveAvgPool1d(1)\nFeature Vector: 128-D", "#FFD166", "#333333"),
                ("Dense Classifier\nFC(128→64) + ReLU + Dropout(0.3)\nFC(64→2) + Softmax", "#C7F9CC", "#1D3557"),
                ("Solar Flare Alert\n[P(Quiet), P(Major Flare)]", "#80ED99", "#004B23"),
            ],
            "highlight": "★ Exponential Receptive Field\n★ Dilated Causal Temporal Convolutions"
        },
        {
            "name": "ResNet1D (Rank 2)",
            "subtitle": "Deep 1D Residual Network",
            "metric": "ROC-AUC: 0.9078 | F1: 0.7477",
            "tag_color": "#3A86FF",
            "header_bg": "#EBF2FF",
            "blocks": [
                ("Input Sequence\n(Batch, 4, 3600)", "#E0E1DD", "#333333"),
                ("Initial Convolution\nConv1d(4→32, k=7, s=2, p=3)\nBatchNorm1d + ReLU\nShape: (Batch, 32, 1800)", "#BEE1E6", "#1D3557"),
                ("Residual Block 1 (s=2)\nConv1d(32→64, k=5) + BN + ReLU\nConv1d(64→64, k=3) + BN\n+ Projected Shortcut (1x1, s=2)", "#D8E2DC", "#1D3557"),
                ("Residual Block 2 (s=2)\nConv1d(64→128, k=5) + BN + ReLU\nConv1d(128→128, k=3) + BN\n+ Projected Shortcut (1x1, s=2)", "#D8E2DC", "#1D3557"),
                ("AdaptiveAvgPool1d(1)\nFeature Vector: 128-D", "#FFD166", "#333333"),
                ("Dense Classifier\nFC(128→32) + ReLU + Dropout(0.3)\nFC(32→2) + Softmax", "#C7F9CC", "#1D3557"),
                ("Solar Flare Alert\n[P(Quiet), P(Major Flare)]", "#80ED99", "#004B23"),
            ],
            "highlight": "★ Deep Residual Skip Pathways\n★ Eliminates Vanishing Gradient"
        },
        {
            "name": "CNN + BiLSTM (Rank 3)",
            "subtitle": "Convolutional Recurrent Hybrid",
            "metric": "ROC-AUC: 0.8800 | F1: 0.7477",
            "tag_color": "#8338EC",
            "header_bg": "#F3EBFF",
            "blocks": [
                ("Input Sequence\n(Batch, 4, 3600)", "#E0E1DD", "#333333"),
                ("Conv Stage 1\nConv1d(4→32, k=5, s=2) + BN\nReLU + MaxPool1d(2)\nShape: (Batch, 32, 900)", "#BEE1E6", "#1D3557"),
                ("Conv Stage 2\nConv1d(32→64, k=3, s=2) + BN\nReLU + MaxPool1d(2)\nShape: (Batch, 64, 225)", "#BEE1E6", "#1D3557"),
                ("2-Layer Bidirectional LSTM\nHidden Dim = 64 (Forward + Backward)\nDropout = 0.2\nOutput Shape: (Batch, 225, 128)", "#E8DFF5", "#4A148C"),
                ("Temporal Mean Pooling\nMean over 225 timesteps\nContext Vector: 128-D", "#FFD166", "#333333"),
                ("Dense Classifier\nFC(128→64) + ReLU + Dropout(0.3)\nFC(64→2) + Softmax", "#C7F9CC", "#1D3557"),
                ("Solar Flare Alert\n[P(Quiet), P(Major Flare)]", "#80ED99", "#004B23"),
            ],
            "highlight": "★ Local Spatial Feature Extraction\n★ Bidirectional Memory Encoding"
        },
        {
            "name": "CNN+Attn+BiLSTM (Rank 4)",
            "subtitle": "Attention-Gated Recurrent",
            "metric": "ROC-AUC: 0.8778 | F1: 0.7477",
            "tag_color": "#FF006E",
            "header_bg": "#FFEBF2",
            "blocks": [
                ("Input Sequence\n(Batch, 4, 3600)", "#E0E1DD", "#333333"),
                ("Conv Front-End\nConv1d(4→32, k=5, s=2) + BN + ReLU\nMaxPool1d(2)\nConv1d(32→64, k=3, s=2) + BN + ReLU", "#BEE1E6", "#1D3557"),
                ("2-Layer Bidirectional LSTM\nHidden Dim = 64 per direction\nOutput Shape: (Batch, 450, 128)", "#E8DFF5", "#4A148C"),
                ("Self-Attention Mechanism\nScore = Linear(128→64) → Tanh → Linear(64→1)\nAttention Weights = Softmax(Score, dim=time)\nWeighted Context Sum: (Batch, 128)", "#FDE2E4", "#9B111E"),
                ("Dense Classifier\nFC(128→64) + ReLU + Dropout(0.3)\nFC(64→2) + Softmax", "#C7F9CC", "#1D3557"),
                ("Solar Flare Alert\n[P(Quiet), P(Major Flare)]", "#80ED99", "#004B23"),
            ],
            "highlight": "★ Soft Temporal Alignment\n★ Dynamic Precursor Focus Weights"
        },
        {
            "name": "InceptionTime (Rank 5)",
            "subtitle": "Multi-Scale Temporal Inception",
            "metric": "ROC-AUC: 0.8703 | F1: 0.7477",
            "tag_color": "#FB5607",
            "header_bg": "#FFF0E6",
            "blocks": [
                ("Input Sequence\n(Batch, 4, 3600)", "#E0E1DD", "#333333"),
                ("Inception Module 1 (Parallel)\n• Conv1d(k=10, ch=10)\n• Conv1d(k=20, ch=10)\n• Conv1d(k=40, ch=10)\n• MaxPool(k=3) + Conv1d(k=1, ch=10)\nConcat + BN + ReLU + MaxPool(2)", "#FEEAFA", "#6A0572"),
                ("Inception Module 2 (Parallel)\n• 1x1 Bottleneck (ch=16)\n• Conv1d(k=10, 20, 40)\n• MaxPool branch + 1x1 Conv\nConcat (84 Channels) + BN + ReLU", "#FEEAFA", "#6A0572"),
                ("AdaptiveAvgPool1d(1)\nFeature Vector: 84-D", "#FFD166", "#333333"),
                ("Dense Classifier\nDropout(0.3)\nFC(84→32) + ReLU\nFC(32→2) + Softmax", "#C7F9CC", "#1D3557"),
                ("Solar Flare Alert\n[P(Quiet), P(Major Flare)]", "#80ED99", "#004B23"),
            ],
            "highlight": "★ Multi-Scale Kernel Ensembles\n★ Captures Short & Long Burst Dynamics"
        }
    ]

    for idx, model in enumerate(models_meta):
        cx = start_x + idx * (col_width + col_gap)

        # Header Box
        header_box = patches.FancyBboxPatch(
            (cx, 84.5), col_width, 7.2,
            boxstyle="round,pad=0.01,rounding_size=0.012",
            facecolor=model["header_bg"],
            edgecolor=model["tag_color"],
            linewidth=2.0,
            zorder=3
        )
        ax.add_patch(header_box)
        ax.text(cx + col_width / 2, 89.6, model["name"], ha="center", va="center", fontsize=11.5, fontweight="bold", color="#1D3557", zorder=4)
        ax.text(cx + col_width / 2, 87.5, model["subtitle"], ha="center", va="center", fontsize=8.8, fontweight="bold", color="#457B9D", zorder=4)
        ax.text(cx + col_width / 2, 85.7, model["metric"], ha="center", va="center", fontsize=8.2, fontweight="bold", color=model["tag_color"], zorder=4)

        # Architectural Highlights Tag at Bottom
        high_box = patches.FancyBboxPatch(
            (cx, 1.5), col_width, 4.5,
            boxstyle="round,pad=0.01,rounding_size=0.01",
            facecolor="#FFFFFF",
            edgecolor=model["tag_color"],
            linewidth=1.2,
            zorder=3
        )
        ax.add_patch(high_box)
        ax.text(cx + col_width / 2, 3.75, model["highlight"], ha="center", va="center", fontsize=7.8, fontweight="bold", color="#1D3557", zorder=4)

        # Sequential Blocks
        num_blocks = len(model["blocks"])
        block_w = col_width - 1.2
        bx = cx + 0.6

        # Vertical range available: 83 down to 7.0 (height = 76)
        total_avail_h = 76.0
        block_h = total_avail_h / (num_blocks * 1.35)
        gap_h = (total_avail_h - (num_blocks * block_h)) / (num_blocks - 1)

        cur_y = 82.5 - block_h

        for b_idx, (b_text, bg_col, text_col) in enumerate(model["blocks"]):
            draw_rounded_box(
                ax, bx, cur_y, block_w, block_h,
                b_text, bg_col,
                border_color="#99A2AA",
                fontsize=8.2,
                text_color=text_col
            )

            # Draw connector arrow to next block
            if b_idx < num_blocks - 1:
                next_y = cur_y - gap_h - block_h
                draw_arrow(
                    ax,
                    bx + block_w / 2, cur_y,
                    bx + block_w / 2, cur_y - gap_h,
                    color=model["tag_color"],
                    lw=1.6
                )
                cur_y = next_y

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Generated: {output_path}")

if __name__ == "__main__":
    generate_architectures_figure("all_models_architecture.png")
    Path("results/plots").mkdir(parents=True, exist_ok=True)
    generate_architectures_figure("results/plots/all_models_architecture.png")
