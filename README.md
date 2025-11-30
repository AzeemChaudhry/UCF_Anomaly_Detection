# UCF-Crime Surveillance Video-Language Understanding: Complete Implementation Pipeline

## **Phase 1: Data Preparation & Preprocessing**

**Chunk 1 - 5 categories handled:**
* Categories: Abuse, Arrest, Arson, Assault, Burglary
* Videos: ~475 videos (~12GB raw → 1-1.5GB features)
* Storage needed: 13GB temporarily (12GB videos + 1GB features)

**Chunk 2 - 5 categories handled:**
* Categories: Explosion, Fighting, Robbery, Shooting, Stealing  
* Videos: ~475 videos (~12GB raw → 1-1.5GB features)
* Storage needed: 13GB temporarily

**Chunk 3 - 4 categories handled:**
* Categories: Shoplifting, Vandalism, Road Accident, Normal
* Videos: ~950 videos (~25GB raw → 2-3GB features)
* Storage needed: 28GB temporarily

## Data Structure 
This dataset contains pre-extracted 1024-dimensional I3D RGB features and frame-level temporal anomaly labels for videos from the UCF-Crime dataset.
It is hosted on the HuggingFace Hub to provide reliable storage and global accessibility, enabling researchers and practitioners to easily integrate it into their video anomaly-detection pipelines.

```
ucf_crime_features_labeled.h5
├── category_name/
│   ├── video_name/
│   │   ├── features [T, 1024]
│   │   ├── labels [T]
│   │   ├── sentences 
│   │   └── attributes:
│   │       ├── duration
│   │       ├── split (Train/Val/Test)
│   │       ├── num_anomaly_segments
│   │       ├── annotation_key
│   │       └── category
```
