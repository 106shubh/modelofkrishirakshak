import torch
import torchvision.transforms as T
from PIL import Image
from cropguard.config import load_config
from cropguard.inference.service import _load_model
from cropguard.inference.novelty import NoveltyDetector

cfg = load_config("configs/demo.yaml"); cfg.eval.novelty.enabled = True
device = torch.device("cpu")
model = _load_model(cfg, device)
ckpt = torch.load("checkpoints/v2s_real/best_model.pth", map_location="cpu", weights_only=False)
detector = NoveltyDetector.from_checkpoint(ckpt, cfg)

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

img1 = Image.open(r"C:\Users\Dell\.gemini\antigravity\brain\3086eaa9-ad21-4f6c-be53-e29ddaa86ca9\.user_uploaded\media_1789681573507.jpg").convert("RGB")
t1 = transform(img1).unsqueeze(0)

img2 = Image.open(r"data\raw\PlantVillage\raw\Background___Not_Plant\bg_0.jpg").convert("RGB")
t2 = transform(img2).unsqueeze(0)

model.eval()
with torch.no_grad():
    feat1 = model(t1)["features"][0].cpu()
    feat2 = model(t2)["features"][0].cpu()

from cropguard.inference.novelty import knn_distance
d1 = knn_distance(feat1.unsqueeze(0), detector.reference, k=5).item()
d2 = knn_distance(feat2.unsqueeze(0), detector.reference, k=5).item()

print(f"Greenhouse Leaf Distance: {d1:.2f}")
print(f"Random Background Distance: {d2:.2f}")
print(f"Original threshold: {detector.threshold:.2f}")
