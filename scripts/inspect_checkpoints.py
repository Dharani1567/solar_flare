import os
import sys
import glob
from pathlib import Path
import torch

sys.path.append(str(Path(".").resolve()))
from scripts.models import (
    Conv1DModel,
    CNNBiLSTMModel,
    CNNAttentionBiLSTMModel,
    TCNModel,
    InceptionTimeModel,
    ResNet1DModel,
    MODEL_REGISTRY
)

print("=" * 80)
print("INSPECTING AND MATCHING ALL MODEL CHECKPOINTS")
print("=" * 80)

# Check all .pt and .pth files
checkpoints = list(Path(".").rglob("*.pt")) + list(Path(".").rglob("*.pth"))
checkpoints = [c for c in checkpoints if ".git" not in c.parts]

models = {
    "1d_cnn": Conv1DModel(in_channels=4, sequence_length=3600, num_classes=2),
    "cnn_bilstm": CNNBiLSTMModel(in_channels=4, hidden_dim=64, num_classes=2),
    "cnn_attention_bilstm": CNNAttentionBiLSTMModel(in_channels=4, hidden_dim=64, num_classes=2),
    "tcn": TCNModel(in_channels=4, num_classes=2),
    "inceptiontime": InceptionTimeModel(in_channels=4, num_classes=2),
    "resnet1d": ResNet1DModel(in_channels=4, num_classes=2),
}

results = []

for c in checkpoints:
    rel_path = str(c.relative_to(Path("."))).replace("\\", "/")
    sz = c.stat().st_size
    try:
        data = torch.load(c, map_location="cpu", weights_only=False)
        state_dict = data if isinstance(data, dict) and not any(k in data for k in ["model_state_dict", "state_dict"]) else data.get("model_state_dict", data.get("state_dict", data))
        if not isinstance(state_dict, dict):
            continue
            
        matched_model = None
        for m_name, model in models.items():
            model_keys = set(model.state_dict().keys())
            ck_keys = set(state_dict.keys())
            
            # Check key match
            if model_keys == ck_keys:
                # Check shape match
                shapes_match = True
                for k in model_keys:
                    if model.state_dict()[k].shape != state_dict[k].shape:
                        shapes_match = False
                        break
                if shapes_match:
                    matched_model = m_name
                    break
                    
        results.append({
            "path": rel_path,
            "size": sz,
            "matched_model": matched_model,
            "key_count": len(state_dict.keys())
        })
    except Exception as e:
        results.append({
            "path": rel_path,
            "size": sz,
            "matched_model": f"Error: {e}",
            "key_count": 0
        })

df_res = pd.DataFrame(results) if 'pd' in dir() else None
print(f"Total weights inspected: {len(results)}")

matches = {}
for r in results:
    m = r["matched_model"]
    if m and not str(m).startswith("Error"):
        matches.setdefault(m, []).append(r["path"])

print("\n--- MATCHED ARCHITECTURES TO CHECKPOINTS ---")
for arch in models.keys():
    found = matches.get(arch, [])
    print(f"\n[{arch}] -> {len(found)} matching checkpoints:")
    for f in found[:6]:
        print(f"  - {f}")
    if len(found) > 6:
        print(f"  ... and {len(found)-6} more")

unmatched = [r for r in results if r["matched_model"] is None]
print(f"\nUnmatched checkpoints: {len(unmatched)}")
for u in unmatched[:10]:
    print(f"  - {u['path']} ({u['key_count']} keys)")
