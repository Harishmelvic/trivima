"""Run this in Colab to download more training data."""
import os
import urllib.request

print("Trying to download full RE10K dataset...")

# Try MIT server
urls = [
    "http://schadenfreude.csail.mit.edu:8000/",
    "https://github.com/google/realestate10k.git",
]

# Method 1: Try wget from MIT
os.system("wget -q http://schadenfreude.csail.mit.edu:8000/ -O /tmp/re10k_index.html 2>&1")
if os.path.exists("/tmp/re10k_index.html") and os.path.getsize("/tmp/re10k_index.html") > 100:
    print("MIT server available!")
    os.system("wget -r -np -nH --cut-dirs=1 -P /content/data/re10k_full http://schadenfreude.csail.mit.edu:8000/ 2>&1 | tail -5")
else:
    print("MIT server not available. Trying Google repo...")
    os.system("git clone https://github.com/google/realestate10k.git /content/data/re10k_poses 2>&1 | tail -3")
    n = len(os.listdir("/content/data/re10k_poses/train")) if os.path.exists("/content/data/re10k_poses/train") else 0
    print("Pose files: %d" % n)
    if n > 0:
        print("Success! Now need to download video frames via yt-dlp")
    else:
        print("Google repo also failed.")
        print("Using existing small dataset for now.")
