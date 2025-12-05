# AU Feature Extraction Results

## 📊 Dataset Overview

This directory contains the extracted Action Unit (AU) and emotion features from 7 video clips.

### Dataset Statistics
- **Total Videos**: 7
- **Total Clips**: 42 (6 clips per video, 1 minute each)
- **Total Frames**: 25,920
- **Average frames per video**: ~3,703 frames
- **Average frames per clip**: ~617 frames

### Video IDs
1. AIMB3084
2. BGLE8278
3. GUTH0227
4. JMOR1353
5. PKGP7582
6. VNBX9754
7. WRSB5786

## 📁 Files

### Main Feature File
- **`combined_au_features.csv`** (7.9 MB)
  - Combined features from all 7 videos
  - 25,920 rows (frames) × 32 columns (features + metadata)

### Column Structure
1. **Identifiers** (5 columns):
   - `sample_id`: Unique identifier (e.g., "AIMB3084_clip1")
   - `video_id`: Video identifier
   - `clip_id`: Clip number (1-6)
   - `frame`: Original frame number
   - `approx_time`: Approximate timestamp

2. **Action Units** (20 columns):
   - AU01, AU02, AU04, AU05, AU06, AU07, AU09, AU10, AU11, AU12
   - AU14, AU15, AU17, AU20, AU23, AU24, AU25, AU26, AU28, AU43
   - Values range from 0.0 to 1.0 (activation intensity)

3. **Emotions** (7 columns):
   - anger, disgust, fear, happiness, sadness, surprise, neutral
   - Values range from 0.0 to 1.0 (probability)

## 📈 Key Findings

### Emotion Distribution
- **Neutral**: 76.32% (dominant emotion in most frames)
- **Happiness**: 10.12%
- **Surprise**: 5.07%
- **Disgust**: 4.08%
- **Sadness**: 1.56%
- **Fear**: 1.41%
- **Anger**: 1.41%

### Most Active Action Units
1. **AU01** (Inner Brow Raiser): 0.630
2. **AU11** (Nasolabial Deepener): 0.615
3. **AU17** (Chin Raiser): 0.512
4. **AU15** (Lip Corner Depressor): 0.416
5. **AU02** (Outer Brow Raiser): 0.414

### Per-Video Emotion Profiles
- **GUTH0227**: Most neutral (99.59% neutral)
- **BGLE8278**: Relatively happy (17.69% happiness)
- **JMOR1353**: Most varied emotions (44.73% neutral, mixed other emotions)
- **WRSB5786**: Highest emotional variation (40.48% neutral, mixed emotions)

## 📊 Visualizations

The `visualizations/` directory contains:

1. **`emotion_analysis.png`**
   - Average emotion intensity across all videos
   - Emotion distribution by video
   - Box plots of emotion probabilities
   - Pie chart of dominant emotions

2. **`au_analysis.png`**
   - Heatmap of AU activation by video
   - AU correlation matrix

3. **`temporal_patterns.png`**
   - How emotions change across clip positions (1-6)
   - Temporal dynamics within the 6-minute videos

4. **`video_comparison.png`**
   - Side-by-side comparison of all 7 videos
   - Emotion trajectories for each clip

5. **`statistics_report.txt`**
   - Detailed statistical summary
   - Mean, std, min, max for all features
   - Per-video breakdowns

## 🔧 Processing Details

### Original Video Parameters
- **Duration**: 6 minutes per video
- **FPS**: 30
- **Total frames**: 10,800 per video
- **Frame stride**: 3 (sampling every 3rd frame)
- **Sampled frames**: ~3,600 per video

### Clip Division
Each 6-minute video was divided into 6 equal clips of approximately 1 minute each:
- Clip 1: Frames 0 - ~600
- Clip 2: Frames ~600 - ~1200
- Clip 3: Frames ~1200 - ~1800
- Clip 4: Frames ~1800 - ~2400
- Clip 5: Frames ~2400 - ~3000
- Clip 6: Frames ~3000 - ~3600

## 🚀 Usage Examples

### Load the features in Python
```python
import pandas as pd

# Load the combined features
df = pd.read_csv('combined_au_features.csv')

# Filter by video
video_df = df[df['video_id'] == 'AIMB3084']

# Filter by clip
clip_df = df[df['sample_id'] == 'AIMB3084_clip1']

# Get all AU features
au_cols = [col for col in df.columns if col.startswith('AU')]
au_features = df[au_cols]

# Get all emotion features
emotion_cols = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']
emotion_features = df[emotion_cols]

# Calculate mean emotions per clip
clip_emotions = df.groupby('sample_id')[emotion_cols].mean()
```

### Aggregate Features
```python
# Mean features per video
video_features = df.groupby('video_id').agg({
    'AU01': 'mean',
    'happiness': 'mean',
    # ... add other features
})

# Mean features per clip
clip_features = df.groupby(['video_id', 'clip_id']).agg({
    'AU01': 'mean',
    'happiness': 'mean',
    # ... add other features
})
```

## 📝 Notes

- Frame stride of 3 means we're sampling at 10 FPS effectively (30 fps / 3 = 10 fps)
- Some videos have slightly different frame counts due to detection variations
- All AU and emotion values are normalized between 0.0 and 1.0
- The `approx_time` column shows the timestamp in MM:SS format

## 🔗 Related Files

- **Source data**: `../AU/*_au_emotions.csv`
- **Extraction script**: `../extract_features.py`
- **Visualization script**: `../visualize_features.py`

---

**Generated**: December 1, 2025
**Processing**: py-feat with frame stride=3

