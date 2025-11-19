# py-feat-application

Facial expression analysis application using Py-Feat to extract Action Units (AUs) and emotions from videos.

## Features

- Extract 20 Action Units from video frames
- Extract 7 emotions (anger, disgust, fear, happiness, sadness, surprise, neutral)
- Process videos at 30fps (configurable)
- **Supports videos of any length** - from seconds to hours
- Output to CSV and NumPy formats
- Comprehensive analysis and visualization tools

## Quick Start

### Extract AUs and Emotions from Video

```bash
python extract_au_emotions.py your_video.mp4
```

**The script automatically handles videos of any duration.** 

Example output for a 6-minute video at 30fps:
- AU data: (10,800 frames × 20 AUs)
- Emotion data: (10,800 frames × 7 emotions)
- Combined data in both CSV and NPY formats

Processing time: ~1-2 seconds per frame on CPU

## Installation

```bash
# Activate the virtual environment
source venv/bin/activate

# Or use the activation script
source activate_pyfeat.sh
```

## Scripts

- `extract_au_emotions.py` - Main script to extract AUs and emotions from video
- `detect_video.py` - Original video detection script
- `analyze_results.py` - Analysis script for results
- `create_annotated_video.py` - Create annotated video with detections

## Documentation

- [USAGE_GUIDE.md](USAGE_GUIDE.md) - Detailed usage guide and examples
- [QUICKSTART.md](QUICKSTART.md) - Quick start guide

## Output Format

### NumPy Arrays (.npy)
- `*_au_data.npy` - Shape: (num_frames, 20) - AU values
- `*_emotion_data.npy` - Shape: (num_frames, 7) - Emotion values
- `*_au_emotions.npy` - Shape: (num_frames, 27) - Combined data

### CSV Files
- `*_au_emotions.csv` - Compact CSV with frame, time, AUs, and emotions
- `*_full_predictions.csv` - Complete predictions with all columns

## References

- [Py-Feat](https://py-feat.org) - Python Facial Expression Analysis Toolbox
- [FACS](https://www.paulekman.com/facial-action-coding-system/) - Facial Action Coding System

