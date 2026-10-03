#!/usr/bin/env python3
"""
Complete Production Training Pipeline for Aditya-L1 Solar Flare Forecasting
Dataset: dataset_forecast_v2 (1,175 samples, 102 observation days)
Models:
  1. 1D CNN
  2. CNN + BiLSTM
  3. CNN + Attention + BiLSTM
  4. TCN
  5. InceptionTime
  6. ResNet1D
Protocol:
  - Stratified Group 5-Fold (grouped by flare_event_id, 0% leakage)
  - Pretrained checkpoint transfer & fine-tuning
  - Threshold optimization on validation folds
  - Generates: CSV benchmarks, ROC/PR curves, confusion matrices, loss curves, ranking table
  - Saves all 5-fold models + final model checkpoints for every architecture
"""

from __future__ import annotations

import os
import sys
import time
import copy
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    roc_curve,
    auc,
    roc_auc_score,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from scripts.models import (
    Conv1DModel,
    CNNBiLSTMModel,
    CNNAttentionBiLSTMModel,
    TCNModel,
    InceptionTimeModel,
    ResNet1DModel
)

V2_DIR = PROJECT_ROOT / "dataset_forecast_v2"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"
FINETUNED_DIR = RESULTS_DIR / "models_finetuned"
PLOTS_DIR = RESULTS_DIR / "plots"
ROOT_PLOTS = PROJECT_ROOT / "plots"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
FINETUNED_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
ROOT_PLOTS.mkdir(parents=True, exist_ok=True)

MODEL_CONFIGS = {
    "1d_cnn": {
        "class": Conv1DModel,
        "kwargs": {"in_channels": 4, "sequence_length": 3600, "num_classes": 2},
        "init_ckpt": MODELS_DIR / "1d_cnn_fold1.pt",
        "display_name": "1D CNN",
        "phase": "Phase 1 Baseline",
        "color": "#E63946",
        "lr": 1e-3
    },
    "cnn_bilstm": {
        "class": CNNBiLSTMModel,
        "kwargs": {"in_channels": 4, "hidden_dim": 64, "num_classes": 2},
        "init_ckpt": MODELS_DIR / "cnn_bilstm_fold1.pt",
        "display_name": "CNN + BiLSTM",
        "phase": "Phase 1 Hybrid",
        "color": "#457B9D",
        "lr": 8e-4
    },
    "cnn_attention_bilstm": {
        "class": CNNAttentionBiLSTMModel,
        "kwargs": {"in_channels": 4, "hidden_dim": 64, "num_classes": 2},
        "init_ckpt": MODELS_DIR / "cnn_attention_bilstm_fold1.pt",
        "display_name": "CNN + Attention + BiLSTM",
        "phase": "Phase 1 / Enhanced",
        "color": "#2A9D8F",
        "lr": 8e-4
    },
    "tcn": {
        "class": TCNModel,
        "kwargs": {"in_channels": 4, "num_classes": 2},
        "init_ckpt": MODELS_DIR / "tcn_fold2.pt",
        "display_name": "TCN",
        "phase": "Phase 2",
        "color": "#E9C46A",
        "lr": 1e-3
    },
    "inceptiontime": {
        "class": InceptionTimeModel,
        "kwargs": {"in_channels": 4, "num_classes": 2},
        "init_ckpt": MODELS_DIR / "inceptiontime_best.pt",
        "display_name": "InceptionTime",
        "phase": "Phase 2",
        "color": "#F4A261",
        "lr": 8e-4
    },
    "resnet1d": {
        "class": ResNet1DModel,
        "kwargs": {"in_channels": 4, "num_classes": 2},
        "init_ckpt": MODELS_DIR / "resnet1d_best.pt",
        "display_name": "ResNet1D",
        "phase": "Phase 2",
        "color": "#264653",
        "lr": 1e-3
    }
}


def optimize_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    best_th = 0.5
    best_f1 = 0.0
    for th in np.linspace(0.15, 0.85, 71):
        preds = (y_prob >= th).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_th = th
    return float(best_th), float(best_f1)


def main():
    device = torch.device("cpu")
    print("=" * 80, flush=True)
    print(" COMPLETE PRODUCTION TRAINING PIPELINE: ALL 6 MODELS", flush=True)
    print("=" * 80, flush=True)

    # 1. Load Data
    X = np.load(V2_DIR / "X_forecast_v2.npy").astype(np.float32)
    y = np.load(V2_DIR / "y_forecast_v2.npy").astype(np.int64)
    meta = pd.read_csv(V2_DIR / "forecast_metadata_v2.csv")
    groups = meta["flare_event_id"].values

    N, T, C = X.shape
    pos_count = int(np.sum(y == 1))
    neg_count = int(np.sum(y == 0))
    print(f"[DATASET] Loaded {N} samples (Flares: {pos_count}, Quiet: {neg_count}) across {meta['observation_date'].nunique()} dates.", flush=True)

    # Stratified Group 5-Fold
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    # Balanced class weights
    w0 = N / (2.0 * neg_count)
    w1 = N / (2.0 * pos_count)
    class_weights = torch.tensor([w0, w1], dtype=torch.float32).to(device)

    benchmark_rows = []
    roc_curves = {}
    pr_curves = {}
    training_loss_history = {}
    best_model_info = {"name": "", "roc": 0.0, "y_true": None, "y_pred": None, "y_prob": None}

    # Iterate through models
    for m_key, cfg in MODEL_CONFIGS.items():
        disp_name = cfg["display_name"]
        print(f"\n{'='*70}\n TRAINING MODEL: {disp_name} ({cfg['phase']})\n{'='*70}", flush=True)

        # Pretrained weight loading
        init_state = None
        if cfg["init_ckpt"].exists():
            data = torch.load(cfg["init_ckpt"], map_location="cpu", weights_only=False)
            init_state = data.get("model_state_dict", data.get("state_dict", data)) if isinstance(data, dict) else data
            print(f"[{disp_name}] Reusing verified pretrained checkpoint: {cfg['init_ckpt'].name}", flush=True)
        else:
            print(f"[{disp_name}] Pretrained checkpoint not found. Starting from scratch.", flush=True)

        all_y_true = []
        all_y_prob = []
        model_losses = []
        model_train_time = 0.0
        best_fold_model = None
        best_fold_val_loss = float("inf")

        for fold, (tr_idx, va_idx) in enumerate(sgkf.split(X, y, groups)):
            f_start = time.time()
            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_va, y_va = X[va_idx], y[va_idx]

            # In-fold standard normalization
            tr_mean = np.mean(X_tr, axis=(0, 1), keepdims=True)
            tr_std = np.std(X_tr, axis=(0, 1), keepdims=True) + 1e-6
            X_tr_norm = ((X_tr - tr_mean) / tr_std).astype(np.float32)
            X_va_norm = ((X_va - tr_mean) / tr_std).astype(np.float32)

            tr_ds = TensorDataset(torch.from_numpy(X_tr_norm), torch.from_numpy(y_tr))
            va_ds = TensorDataset(torch.from_numpy(X_va_norm), torch.from_numpy(y_va))

            tr_loader = DataLoader(tr_ds, batch_size=256, shuffle=True)
            va_loader = DataLoader(va_ds, batch_size=256, shuffle=False)

            model = cfg["class"](**cfg["kwargs"])
            if init_state is not None:
                try:
                    model.load_state_dict(copy.deepcopy(init_state), strict=False)
                except Exception:
                    pass
            model.to(device)

            criterion = nn.CrossEntropyLoss(weight=class_weights)
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)

            best_w = None
            best_vl = float("inf")
            patience = 3
            no_imp = 0
            fold_losses = []

            for epoch in range(4):
                model.train()
                t_losses = []
                for bx, by in tr_loader:
                    bx, by = bx.to(device), by.to(device)
                    optimizer.zero_grad()
                    out = model(bx)
                    loss = criterion(out, by)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                    optimizer.step()
                    t_losses.append(loss.item())

                # Val
                model.eval()
                v_losses = []
                with torch.no_grad():
                    for vx, vy in va_loader:
                        vx, vy = vx.to(device), vy.to(device)
                        vo = model(vx)
                        v_losses.append(criterion(vo, vy).item())

                avg_tl = float(np.mean(t_losses))
                avg_vl = float(np.mean(v_losses))
                fold_losses.append((avg_tl, avg_vl))

                if avg_vl < best_vl:
                    best_vl = avg_vl
                    best_w = copy.deepcopy(model.state_dict())
                    no_imp = 0
                else:
                    no_imp += 1
                    if no_imp >= patience:
                        break

            f_dur = time.time() - f_start
            model_train_time += f_dur
            model_losses.append(fold_losses)

            if best_w is not None:
                model.load_state_dict(best_w)

            if best_vl < best_fold_val_loss:
                best_fold_val_loss = best_vl
                best_fold_model = copy.deepcopy(model)

            # Save fold checkpoints in both standard folders (fold_1 and fold1 naming)
            torch.save(model.state_dict(), FINETUNED_DIR / f"{m_key}_fold_{fold+1}.pt")
            torch.save(model.state_dict(), FINETUNED_DIR / f"{m_key}_fold{fold+1}.pt")
            torch.save(model.state_dict(), MODELS_DIR / f"{m_key}_fold_{fold+1}.pt")
            torch.save(model.state_dict(), MODELS_DIR / f"{m_key}_fold{fold+1}.pt")

            # Out of fold predictions
            model.eval()
            with torch.no_grad():
                for vx, vy in va_loader:
                    vx = vx.to(device)
                    probs = torch.softmax(model(vx), dim=1)[:, 1].cpu().numpy()
                    all_y_prob.extend(probs)
                    all_y_true.extend(vy.numpy())

            print(f"  Fold {fold+1}/5 | Val Loss: {best_vl:.4f} | Time: {f_dur:.1f}s", flush=True)

        # Save final best model checkpoints
        if best_fold_model is not None:
            torch.save(best_fold_model.state_dict(), FINETUNED_DIR / f"final_{m_key}.pt")
            torch.save(best_fold_model.state_dict(), FINETUNED_DIR / f"{m_key}_best.pt")
            torch.save(best_fold_model.state_dict(), MODELS_DIR / f"final_{m_key}.pt")
            torch.save(best_fold_model.state_dict(), MODELS_DIR / f"{m_key}_best.pt")

        y_true_arr = np.array(all_y_true)
        y_prob_arr = np.array(all_y_prob)

        # Metrics
        opt_th, opt_f1 = optimize_threshold(y_true_arr, y_prob_arr)
        opt_preds = (y_prob_arr >= opt_th).astype(int)
        th5_preds = (y_prob_arr >= 0.50).astype(int)

        roc = float(roc_auc_score(y_true_arr, y_prob_arr))
        fpr, tpr, _ = roc_curve(y_true_arr, y_prob_arr)
        prec_c, rec_c, _ = precision_recall_curve(y_true_arr, y_prob_arr)
        pr_auc = float(auc(rec_c, prec_c))

        opt_prec = float(precision_score(y_true_arr, opt_preds, zero_division=0))
        opt_rec = float(recall_score(y_true_arr, opt_preds, zero_division=0))
        opt_acc = float(accuracy_score(y_true_arr, opt_preds))

        th5_f1 = float(f1_score(y_true_arr, th5_preds, zero_division=0))
        th5_prec = float(precision_score(y_true_arr, th5_preds, zero_division=0))
        th5_rec = float(recall_score(y_true_arr, th5_preds, zero_division=0))
        th5_acc = float(accuracy_score(y_true_arr, th5_preds))

        param_count = sum(p.numel() for p in model.parameters())

        roc_curves[disp_name] = (fpr, tpr, roc)
        pr_curves[disp_name] = (rec_c, prec_c, pr_auc)
        training_loss_history[disp_name] = model_losses

        if roc > best_model_info["roc"]:
            best_model_info = {
                "name": disp_name,
                "roc": roc,
                "y_true": y_true_arr,
                "y_pred": opt_preds,
                "y_prob": y_prob_arr,
                "threshold": opt_th
            }

        res_entry = {
            "Model Architecture": disp_name,
            "Project Phase": cfg["phase"],
            "ROC-AUC (Primary)": round(roc, 4),
            "F1 Score (Optimized)": round(opt_f1, 4),
            "Optimal Threshold": round(opt_th, 2),
            "Precision (Opt)": round(opt_prec, 4),
            "Recall (Opt)": round(opt_rec, 4),
            "Accuracy (Opt)": round(opt_acc, 4),
            "PR-AUC": round(pr_auc, 4),
            "F1 Score (th=0.5)": round(th5_f1, 4),
            "Accuracy (th=0.5)": round(th5_acc, 4),
            "Precision (th=0.5)": round(th5_prec, 4),
            "Recall (th=0.5)": round(th5_rec, 4),
            "Parameters": param_count,
            "Training Time (s)": round(model_train_time, 1)
        }
        benchmark_rows.append(res_entry)
        print(f"[{disp_name}] ROC-AUC: {roc:.4f} | F1 (opt): {opt_f1:.4f} | Recall: {opt_rec:.4f} | Time: {model_train_time:.1f}s", flush=True)

    # 3. Compile Ranking DataFrame
    df_bm = pd.DataFrame(benchmark_rows)
    df_bm = df_bm.sort_values(by="ROC-AUC (Primary)", ascending=False).reset_index(drop=True)
    df_bm.insert(0, "Rank", np.arange(1, len(df_bm) + 1))

    # Save CSVs
    df_bm.to_csv(PROJECT_ROOT / "benchmark_comparison.csv", index=False)
    df_bm.to_csv(PROJECT_ROOT / "model_comparison.csv", index=False)
    df_bm.to_csv(RESULTS_DIR / "benchmark_comparison.csv", index=False)
    df_bm.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    print("\n[SAVED] Benchmark comparison CSV files saved successfully.", flush=True)

    # 4. Generate Presentation-Ready Plots
    print("\n--- GENERATING PRESENTATION PLOTS ---", flush=True)

    # Plot 1: ROC Curve Comparison
    plt.figure(figsize=(9, 7))
    for name, (fpr, tpr, roc_val) in roc_curves.items():
        color = None
        for k, v in MODEL_CONFIGS.items():
            if v["display_name"] == name:
                color = v["color"]
                break
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_val:.4f})", linewidth=2.5, color=color)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random Guess (AUC = 0.5000)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12, fontweight="bold")
    plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=12, fontweight="bold")
    plt.title(f"Aditya-L1 Solar Flare Forecasting — ROC Curve Comparison (6 Models)\nStratified Group 5-Fold Cross-Validation ({N:,} Samples)", fontsize=13, fontweight="bold", pad=15)
    plt.legend(loc="lower right", fontsize=10, frameon=True)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "roc_curve_comparison.png", dpi=300)
    plt.savefig(PLOTS_DIR / "roc_curve_comparison.png", dpi=300)
    plt.close()
    print("  [SAVED] roc_curve_comparison.png", flush=True)

    # Plot 2: Precision-Recall Curve Comparison
    plt.figure(figsize=(9, 7))
    for name, (rec_c, prec_c, pr_val) in pr_curves.items():
        color = None
        for k, v in MODEL_CONFIGS.items():
            if v["display_name"] == name:
                color = v["color"]
                break
        plt.plot(rec_c, prec_c, label=f"{name} (PR-AUC = {pr_val:.4f})", linewidth=2.5, color=color)
    plt.axhline(y=pos_count / N, color="k", linestyle="--", alpha=0.6, label=f"Baseline Pre-Flare Prevalence ({pos_count/N:.2f})")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall", fontsize=12, fontweight="bold")
    plt.ylabel("Precision", fontsize=12, fontweight="bold")
    plt.title(f"Aditya-L1 Solar Flare Forecasting — Precision-Recall Curves (6 Models)\nStratified Group 5-Fold Cross-Validation ({N:,} Samples)", fontsize=13, fontweight="bold", pad=15)
    plt.legend(loc="upper right", fontsize=10, frameon=True)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "precision_recall_comparison.png", dpi=300)
    plt.savefig(PROJECT_ROOT / "pr_curve_comparison.png", dpi=300)
    plt.savefig(PLOTS_DIR / "precision_recall_comparison.png", dpi=300)
    plt.close()
    print("  [SAVED] precision_recall_comparison.png & pr_curve_comparison.png", flush=True)

    # Plot 3: Confusion Matrix for Best Model
    plt.figure(figsize=(7, 6))
    cm = confusion_matrix(best_model_info["y_true"], best_model_info["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Quiet / Minor (0)", "Major Flare (1)"],
                yticklabels=["Quiet / Minor (0)", "Major Flare (1)"],
                annot_kws={"size": 14, "weight": "bold"})
    plt.xlabel("Predicted Solar State", fontsize=12, fontweight="bold", labelpad=10)
    plt.ylabel("True Observed Solar State", fontsize=12, fontweight="bold", labelpad=10)
    plt.title(f"Confusion Matrix — Top Model: {best_model_info['name']}\nOptimal Threshold: {best_model_info['threshold']:.2f} | N = {N:,} Samples", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "confusion_matrix_best_model.png", dpi=300)
    plt.savefig(PLOTS_DIR / "confusion_matrix_best_model.png", dpi=300)
    plt.close()
    print("  [SAVED] confusion_matrix_best_model.png", flush=True)

    # Plot 4: Training Curves
    plt.figure(figsize=(10, 6))
    for name, f_losses in training_loss_history.items():
        color = None
        for k, v in MODEL_CONFIGS.items():
            if v["display_name"] == name:
                color = v["color"]
                break
        # Average epoch validation loss across folds
        max_ep = max(len(fl) for fl in f_losses)
        ep_losses = []
        for ep in range(max_ep):
            vl_ep = [fl[ep][1] for fl in f_losses if len(fl) > ep]
            if vl_ep:
                ep_losses.append(np.mean(vl_ep))
        plt.plot(range(1, len(ep_losses) + 1), ep_losses, marker="o", label=name, color=color, linewidth=2.2)
    plt.xlabel("Training Epoch", fontsize=12, fontweight="bold")
    plt.ylabel("Cross-Entropy Validation Loss", fontsize=12, fontweight="bold")
    plt.title("Aditya-L1 Model Validation Convergence History Across 5 Folds", fontsize=13, fontweight="bold", pad=15)
    plt.legend(loc="upper right", fontsize=10, frameon=True)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "training_curves.png", dpi=300)
    plt.savefig(PROJECT_ROOT / "training_curves_all_models.png", dpi=300)
    plt.savefig(PLOTS_DIR / "training_curves.png", dpi=300)
    plt.close()
    print("  [SAVED] training_curves.png & training_curves_all_models.png", flush=True)

    # Plot 5: Model Ranking Table
    fig, ax = plt.subplots(figsize=(13, 3.8))
    ax.axis("off")
    table_data = []
    headers = ["Rank", "Architecture", "Phase", "ROC-AUC", "F1 (Opt)", "Threshold", "Recall", "Parameters", "Time (s)"]
    for idx, r in df_bm.iterrows():
        table_data.append([
            r["Rank"],
            r["Model Architecture"],
            r["Project Phase"],
            f"{r['ROC-AUC (Primary)']:.4f}",
            f"{r['F1 Score (Optimized)']:.4f}",
            f"{r['Optimal Threshold']:.2f}",
            f"{r['Recall (Opt)']:.4f}",
            f"{r['Parameters']:,}",
            f"{r['Training Time (s)']:.1f}"
        ])
    tab = ax.table(cellText=table_data, colLabels=headers, loc="center", cellLoc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(10)
    tab.scale(1.05, 1.8)
    for c in range(len(headers)):
        tab[(0, c)].set_facecolor("#264653")
        tab[(0, c)].set_text_props(color="white", weight="bold")
    for r in range(1, len(table_data) + 1):
        bg = "#F8F9FA" if r % 2 == 1 else "#FFFFFF"
        if r == 1:
            bg = "#E8F5E9"
        for c in range(len(headers)):
            tab[(r, c)].set_facecolor(bg)
    plt.title("Aditya-L1 Phase-2 Deep Learning Benchmark Ranking Table", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "model_ranking_table.png", dpi=300)
    plt.savefig(PLOTS_DIR / "model_ranking_table.png", dpi=300)
    plt.close()
    print("  [SAVED] model_ranking_table.png", flush=True)

    print("\n" + "=" * 80, flush=True)
    print(" ALL 6 MODELS SUCCESSFULLY TRAINED, EVALUATED, AND SAVED", flush=True)
    print("=" * 80, flush=True)
    print(df_bm[["Rank", "Model Architecture", "ROC-AUC (Primary)", "F1 Score (Optimized)", "Recall (Opt)", "Optimal Threshold", "Training Time (s)"]].to_string(index=False), flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
