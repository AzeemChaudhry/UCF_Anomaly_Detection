"""
Convert your HDF5 to PDVC format
"""
import h5py
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from huggingface_hub import hf_hub_download

def prepare_uca_for_pdvc():
    print("="*70)
    print("PREPARING UCA DATASET FOR PDVC")
    print("="*70)
    
    # Download HDF5
    print("\n[1/3] Downloading HDF5 from HuggingFace...")
    repo_id = "Rahima411/ucf-anomaly-detection-mapped"
    hdf5_path = hf_hub_download(
        repo_id=repo_id,
        filename="ucf_crime_features_labeled.h5",
        repo_type="dataset"
    )
    print(f"  ✓ Downloaded to: {hdf5_path}")
    
    # Create directories
    print("\n[2/3] Creating directories...")
    Path('data/uca/features/i3d').mkdir(parents=True, exist_ok=True)
    Path('data/uca/captiondata').mkdir(parents=True, exist_ok=True)
    print("  ✓ Directories created")
    
    # Extract and convert
    print("\n[3/3] Extracting features and annotations...")
    
    train_ann = {}
    val_ann = {}
    test_ann = {}
    
    with h5py.File(hdf5_path, 'r') as h5f:
        total = sum(len(list(h5f[cat].keys())) for cat in h5f.keys())
        
        with tqdm(total=total, desc="Processing videos") as pbar:
            for category in h5f.keys():
                for video_name in h5f[category].keys():
                    vid = h5f[category][video_name]
                    
                    # Extract features
                    features = np.array(vid['features'])
                    video_id = f"{category}_{video_name}"
                    np.save(f'data/uca/features/i3d/{video_id}.npy', features)
                    
                    # Extract annotations
                    duration = float(vid.attrs['duration'])
                    split = vid.attrs['split']
                    if isinstance(split, bytes):
                        split = split.decode('utf-8')
                    
                    # Get sentences
                    if 'sentences' in vid:
                        sentences_raw = vid['sentences'][:]
                        sentences = [s.decode('utf-8') if isinstance(s, bytes) else s 
                                   for s in sentences_raw]
                    else:
                        sentences = []
                    
                    # Convert labels to timestamps
                    labels = np.array(vid['labels'])
                    timestamps = []
                    
                    if len(labels) > 0:
                        in_segment = False
                        start_idx = 0
                        
                        for i, label in enumerate(labels):
                            if label == 1 and not in_segment:
                                start_idx = i
                                in_segment = True
                            elif label == 0 and in_segment:
                                start_time = (start_idx / len(labels)) * duration
                                end_time = ((i-1) / len(labels)) * duration
                                timestamps.append([start_time, end_time])
                                in_segment = False
                        
                        if in_segment:
                            start_time = (start_idx / len(labels)) * duration
                            timestamps.append([start_time, duration])
                    
                    # Create PDVC annotation
                    annotation = {
                        'duration': duration,
                        'timestamps': timestamps,
                        'sentences': sentences if sentences else [f"event {i+1}" for i in range(len(timestamps))]
                    }
                    
                    # Add to appropriate split
                    if split.lower() == 'train':
                        train_ann[video_id] = annotation
                    elif split.lower() == 'val':
                        val_ann[video_id] = annotation
                    elif split.lower() == 'test':
                        test_ann[video_id] = annotation
                    
                    pbar.update(1)
    
    # Save annotations
    print("\n[4/4] Saving annotations...")
    with open('data/uca/captiondata/train.json', 'w') as f:
        json.dump(train_ann, f, indent=2)
    print(f"  ✓ train.json: {len(train_ann)} videos")
    
    with open('data/uca/captiondata/val.json', 'w') as f:
        json.dump(val_ann, f, indent=2)
    print(f"  ✓ val.json: {len(val_ann)} videos")
    
    with open('data/uca/captiondata/test.json', 'w') as f:
        json.dump(test_ann, f, indent=2)
    print(f"  ✓ test.json: {len(test_ann)} videos")
    
    # Build vocabulary
    from collections import Counter
    word_counts = Counter()
    for ann in train_ann.values():
        for sent in ann['sentences']:
            word_counts.update(sent.lower().split())
    
    vocab = {'<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<UNK>': 3}
    idx = 4
    for word, count in word_counts.most_common():
        if count >= 5:
            vocab[word] = idx
            idx += 1
    
    with open('data/uca/captiondata/vocabulary.json', 'w') as f:
        json.dump(vocab, f, indent=2)
    print(f"  ✓ vocabulary.json: {len(vocab)} words")
    
    print("\n" + "="*70)
    print("PREPARATION COMPLETE!")
    print("="*70)
    print(f"\nStatistics:")
    print(f"  Train: {len(train_ann)} | Val: {len(val_ann)} | Test: {len(test_ann)}")
    print(f"  Total: {len(train_ann) + len(val_ann) + len(test_ann)} videos")
    print("\n✓ Ready for PDVC training!")

if __name__ == "__main__":
    prepare_uca_for_pdvc()