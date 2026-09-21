import os
import urllib.request

out_dir = "data/raw/PlantVillage/raw/Background___Not_Plant"
os.makedirs(out_dir, exist_ok=True)

for i in range(50):
    url = f"https://picsum.photos/256/256?random={i}"
    try:
        urllib.request.urlretrieve(url, os.path.join(out_dir, f"bg_{i}.jpg"))
        print(f"Downloaded {i}")
    except Exception as e:
        print(f"Error {i}: {e}")
