import torch
from cropguard.config import load_config
from cropguard.models.multimodal import MultimodalClassifier

cfg = load_config("configs/demo.yaml")
ckpt = torch.load("checkpoints/v2s_real/best_model.pth", map_location="cpu", weights_only=False)

# Create 39 class model
model = MultimodalClassifier(num_classes=39, backbone="efficientnet_v2_s", pretrained=False)

old_state = ckpt["model_state_dict"]
new_state = model.state_dict()

for k, v in old_state.items():
    if "head." in k:  # e.g., head.weight, head.bias
        # Copy the first 38 classes
        new_state[k][:38] = v
        # Initialize the 39th class to 0 or random
        if 'weight' in k:
            torch.nn.init.normal_(new_state[k][38:], std=0.01)
        else:
            torch.nn.init.zeros_(new_state[k][38:])
    elif "metadata.crop_emb.weight" in k:
        # Copy the first 15 crops
        new_state[k][:15] = v
        # 16th crop (id 99 mapped to something or just random)
        torch.nn.init.normal_(new_state[k][15:], std=0.01)
    else:
        new_state[k] = v

model.load_state_dict(new_state)

ckpt["model_state_dict"] = new_state
# Also update the cfg inside ckpt if any
torch.save(ckpt, "checkpoints/v2s_real/best_model_39.pth")
print("Saved 39-class checkpoint!")
