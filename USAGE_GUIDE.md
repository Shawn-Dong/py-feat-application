# AU and Emotion Extraction Guide

## Quick Start

Extract Action Units (AUs) and emotions from a video at 30fps:

```bash
python extract_au_emotions.py your_video.mp4
```

## Usage Examples

### 1. Basic Usage (Automatic Output Naming)
```bash
python extract_au_emotions.py shawn.mp4
```
This will create output files with prefix `shawn_`:
- `shawn_au_data.npy` - NumPy array of AU values (shape: [num_frames, 20])
- `shawn_emotion_data.npy` - NumPy array of emotion values (shape: [num_frames, 7])
- `shawn_au_emotions.npy` - Combined array (shape: [num_frames, 27])
- `shawn_au_emotions.csv` - Compact CSV with frame, time, AUs, emotions
- `shawn_full_predictions.csv` - Full CSV with all detection data
- `shawn_column_names.txt` - Reference for column names

### 2. Custom Output Prefix
```bash
python extract_au_emotions.py input.mp4 --output my_results
```
Output files will be named: `my_results_*`

### 3. Different FPS
```bash
python extract_au_emotions.py video.mp4 --fps 24
```

## Output Data Format

### Action Units (AUs)
The script extracts 20 Action Units per frame:
- AU01, AU02, AU04, AU05, AU06, AU07, AU09, AU10
- AU11, AU12, AU14, AU15, AU17, AU20, AU23, AU24
- AU25, AU26, AU28, AU43

### Emotions
The script extracts 7 emotions per frame:
- anger
- disgust
- fear
- happiness
- sadness
- surprise
- neutral

### Data Dimensions for 6-Minute Video
For a 6-minute video at 30fps:
- Expected frames: 30 fps × 6 minutes × 60 seconds = **10,800 frames**
- AU data: `(10800, 20)` - 10,800 frames × 20 AUs
- Emotion data: `(10800, 7)` - 10,800 frames × 7 emotions
- Combined data: `(10800, 27)` - 10,800 frames × 27 features

## Loading Data in Python

### Load NumPy Arrays
```python
import numpy as np

# Load AU data
au_data = np.load('shawn_au_data.npy')
print(f"AU data shape: {au_data.shape}")  # (num_frames, 20)

# Load emotion data
emotion_data = np.load('shawn_emotion_data.npy')
print(f"Emotion data shape: {emotion_data.shape}")  # (num_frames, 7)

# Load combined data
combined_data = np.load('shawn_au_emotions.npy')
print(f"Combined data shape: {combined_data.shape}")  # (num_frames, 27)

# Access specific frame
frame_idx = 0
print(f"Frame {frame_idx} AUs: {au_data[frame_idx]}")
print(f"Frame {frame_idx} emotions: {emotion_data[frame_idx]}")
```

### Load CSV Files
```python
import pandas as pd

# Load compact CSV
df = pd.read_csv('shawn_au_emotions.csv')
print(df.head())

# Access specific columns
print(df['AU01'].values)  # Get AU01 values for all frames
print(df['happiness'].values)  # Get happiness values for all frames

# Filter by time
time_range = df[(df['approx_time'] >= 10) & (df['approx_time'] <= 20)]
print(time_range)
```

### Analyze Specific Time Ranges
```python
import numpy as np
import pandas as pd

# Load data
df = pd.read_csv('shawn_au_emotions.csv')

# Get data for first minute (0-60 seconds)
first_minute = df[df['approx_time'] <= 60]

# Calculate mean AU values for first minute
au_cols = [col for col in df.columns if col.startswith('AU')]
mean_aus = first_minute[au_cols].mean()
print("Mean AU values in first minute:")
print(mean_aus)

# Find frames with highest happiness
happiest_frames = df.nlargest(10, 'happiness')
print("\nTop 10 happiest frames:")
print(happiest_frames[['frame', 'approx_time', 'happiness']])
```

## Understanding the Data

### Action Unit Values
- Range: Typically 0-5 (intensity scale)
- Higher values indicate stronger activation of that AU
- Example: AU12 (lip corner puller) → smiling

### Emotion Values
- Range: 0-1 (probability/confidence)
- Sum across all emotions may not equal 1.0
- Values represent confidence for each emotion category

## Processing Time

For a 6-minute video at 30fps:
- Expected processing time: 15-30 minutes (CPU)
- Frames to process: ~10,800 frames
- Progress is shown during processing

## Troubleshooting

### Video Not Found
```bash
# Make sure the video path is correct
ls -lh your_video.mp4

# Use absolute path if needed
python extract_au_emotions.py /full/path/to/video.mp4
```

### Out of Memory
If processing fails due to memory:
```bash
# Process at lower fps
python extract_au_emotions.py video.mp4 --fps 15
```

### Fewer Frames Than Expected
- Video may be shorter than 6 minutes
- Video may be at different fps (24fps, 29.97fps, etc.)
- Check the console output for actual frame count and fps

## Advanced Usage

### Batch Processing Multiple Videos
```bash
# Create a simple batch script
for video in *.mp4; do
    python extract_au_emotions.py "$video"
done
```

### Custom Analysis
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
au_data = np.load('shawn_au_data.npy')
emotion_data = np.load('shawn_emotion_data.npy')
df = pd.read_csv('shawn_au_emotions.csv')

# Plot emotion timeline
plt.figure(figsize=(12, 6))
plt.plot(df['approx_time'], df['happiness'], label='Happiness')
plt.plot(df['approx_time'], df['sadness'], label='Sadness')
plt.plot(df['approx_time'], df['anger'], label='Anger')
plt.xlabel('Time (seconds)')
plt.ylabel('Emotion Intensity')
plt.title('Emotion Timeline')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('emotion_timeline.png', dpi=150, bbox_inches='tight')
plt.show()

# Analyze AU correlations
import seaborn as sns

au_cols = [col for col in df.columns if col.startswith('AU')]
au_corr = df[au_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(au_corr, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5)
plt.title('Action Unit Correlations')
plt.tight_layout()
plt.savefig('au_correlations.png', dpi=150, bbox_inches='tight')
plt.show()
```

## References

- Py-Feat Documentation: https://py-feat.org
- Action Units Reference: https://www.paulekman.com/facial-action-coding-system/
- FACS (Facial Action Coding System): Standard for describing facial movements

