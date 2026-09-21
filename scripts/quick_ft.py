import torch
from torch import nn
from PIL import Image
from glob import glob
from cropguard.config import load_config
from cropguard.models.multimodal import MultimodalClassifier
import torchvision.transforms as T

# Load
cfg = load_config("configs/demo.yaml")
ckpt = torch.load("checkpoints/v2s_real/best_model_39.pth", map_location="cpu", weights_only=False)
model = MultimodalClassifier(num_classes=39, backbone="efficientnet_v2_s", pretrained=False)
model.load_state_dict(ckpt["model_state_dict"])
model.train()

# Freeze backbone
for param in model.backbone.parameters():
    param.requires_grad = False

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

bg_files = glob("data/raw/PlantVillage/raw/Background___Not_Plant/*.jpg")
print(f"Found {len(bg_files)} background images.")

# Quick 5 iterations on the same batch to overfit it
for i in range(5):
    batch_tensors = []
    for f in bg_files[:16]:
        img = Image.open(f).convert("RGB")
        batch_tensors.append(transform(img))
    
    x = torch.stack(batch_tensors)
    # Target is index 38 (the 39th class)
    targets = torch.full((len(x),), 38, dtype=torch.long)
    
    # Dummy context
    crops = torch.full((len(x),), 15, dtype=torch.long)
    stages = torch.zeros((len(x),), dtype=torch.long)
    regions = torch.zeros((len(x),), dtype=torch.long)
    weather = torch.zeros((len(x), 4), dtype=torch.float32)
    
    optimizer.zero_grad()
    out = model(x, crops, stages, regions, weather)["logits"]
    loss = criterion(out, targets)
    loss.backward()
    optimizer.step()
    
    print(f"Iteration {i} Loss: {loss.item():.4f}")

ckpt["model_state_dict"] = model.state_dict()
torch.save(ckpt, "checkpoints/v2s_real/best_model_39_ft.pth")
print("Saved best_model_39_ft.pth!")
