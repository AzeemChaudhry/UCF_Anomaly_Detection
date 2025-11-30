import json
import numpy as np
from pathlib import Path

print("VERIFICATION")
print("="*60)

# Check features
features = list(Path('data/uca/features/i3d').glob('*.npy'))
print(f"✓ Features: {len(features)} files")

# Check annotations
with open('data/uca/captiondata/train.json') as f:
    train = json.load(f)
with open('data/uca/captiondata/val.json') as f:
    val = json.load(f)
with open('data/uca/captiondata/test.json') as f:
    test = json.load(f)

print(f"✓ Train: {len(train)} videos")
print(f"✓ Val: {len(val)} videos")
print(f"✓ Test: {len(test)} videos")

# Sample check
sample_id = list(train.keys())[0]
print(f"\nSample video: {sample_id}")
print(f"  Duration: {train[sample_id]['duration']:.1f}s")
print(f"  Events: {len(train[sample_id]['timestamps'])}")
print(f"  Captions: {len(train[sample_id]['sentences'])}")

print("\n✅ Setup verified! Ready to train.")