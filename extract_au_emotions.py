#!/usr/bin/env python
"""
Extract Action Units (AUs) and Emotions from Video at 30fps
This script processes videos of any length and extracts AU values and emotions for each frame.
Outputs: CSV and NPY files with AU values (20 AUs) and emotions (7 emotions).

The script can handle videos of any duration - from seconds to hours.
Processing time is approximately 1-2 seconds per frame on CPU.
"""

import os
import sys
import numpy as np
from feat import Detector
import argparse


def extract_au_emotions(video_path, output_prefix=None, target_fps=30):
    """
    Extract Action Units and Emotions from video at specified fps.
    
    Works with videos of any length (seconds to hours).
    
    Parameters:
    -----------
    video_path : str
        Path to input video file (any duration)
    output_prefix : str, optional
        Prefix for output files. If None, uses video filename
    target_fps : int, optional
        Target frames per second for processing (default: 30)
        
    Returns:
    --------
    dict : Dictionary containing AU data, emotion data, and full predictions
    """
    
    print("=" * 70)
    print("AU and Emotion Extraction from Video")
    print("=" * 70)
    
    # Validate input video
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Set output prefix
    if output_prefix is None:
        output_prefix = os.path.splitext(os.path.basename(video_path))[0]
    
    # Create results directory
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"\n✓ Created results directory: {results_dir}")
    
    # Update output prefix to include results directory
    output_prefix = os.path.join(results_dir, output_prefix)
    
    # Initialize detector
    print("\n1. Initializing Py-Feat Detector (using CPU)...")
    detector = Detector(device='cpu')
    print(f"   ✓ Detector ready")
    print(f"   - Face model: {detector.info['face_model']}")
    print(f"   - AU model: {detector.info['au_model']}")
    print(f"   - Emotion model: {detector.info['emotion_model']}")
    
    # Get video info
    print(f"\n2. Processing video: {video_path}")
    
    # Calculate skip_frames based on target fps
    # Note: skip_frames=0 causes a bug in py-feat, use skip_frames=1 for every frame
    # skip_frames=1 means process every frame
    # skip_frames=2 means process every other frame, etc.
    import cv2
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / video_fps if video_fps > 0 else 0
    cap.release()
    
    # Calculate skip_frames to achieve target fps
    if video_fps >= target_fps:
        skip_frames = max(1, int(round(video_fps / target_fps)))
    else:
        skip_frames = 1  # Video is already slower than target fps
    
    print(f"   - Video FPS: {video_fps:.2f}")
    print(f"   - Video duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    print(f"   - Total frames in video: {frame_count}")
    print(f"   - Target FPS: {target_fps}")
    print(f"   - Skip frames: {skip_frames} (process every {skip_frames} frame(s))")
    expected_output_frames = frame_count // skip_frames
    print(f"   - Expected output frames: ~{expected_output_frames}")
    print("   Note: This may take several minutes for long videos...")
    
    # Detect facial expressions in the video
    try:
        video_prediction = detector.detect_video(
            video_path,
            skip_frames=skip_frames,
            face_detection_threshold=0.5,  # More lenient threshold
            output_size=None  # Keep original frame size
        )
    except Exception as e:
        print(f"\n✗ Error during video processing: {e}")
        raise
    
    print(f"\n3. Detection complete!")
    print(f"   - Processed {len(video_prediction)} frames")
    print(f"   - Total columns: {video_prediction.shape[1]}")
    
    # Get video metadata
    if len(video_prediction) > 0:
        first_row = video_prediction.iloc[0]
        frame_height = first_row.get('FrameHeight', 'N/A')
        frame_width = first_row.get('FrameWidth', 'N/A')
        print(f"   - Frame size: {frame_width}x{frame_height}")
        
        # Calculate actual fps from the data
        if 'approx_time' in video_prediction.columns:
            try:
                total_time = float(video_prediction['approx_time'].max())
                actual_fps = len(video_prediction) / total_time if total_time > 0 else 0
                print(f"   - Video duration: {total_time:.2f} seconds")
                print(f"   - Actual FPS: {actual_fps:.2f}")
            except (ValueError, TypeError):
                print(f"   - Video duration: N/A")
                print(f"   - Actual FPS: N/A")
    
    # Extract AU columns (all columns starting with 'AU')
    au_columns = [col for col in video_prediction.columns if col.startswith('AU')]
    print(f"\n4. Action Units detected: {len(au_columns)} AUs")
    print(f"   AUs: {', '.join(au_columns)}")
    
    # Extract emotion columns
    emotion_columns = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']
    available_emotions = [col for col in emotion_columns if col in video_prediction.columns]
    print(f"\n5. Emotions detected: {len(available_emotions)} emotions")
    print(f"   Emotions: {', '.join(available_emotions)}")
    
    # Extract AU data
    au_data = video_prediction[au_columns].values
    print(f"\n6. AU data shape: {au_data.shape}")
    print(f"   - Frames: {au_data.shape[0]}")
    print(f"   - AUs per frame: {au_data.shape[1]}")
    
    # Extract emotion data
    emotion_data = video_prediction[available_emotions].values
    print(f"\n7. Emotion data shape: {emotion_data.shape}")
    print(f"   - Frames: {emotion_data.shape[0]}")
    print(f"   - Emotions per frame: {emotion_data.shape[1]}")
    
    # Combine AU and emotion data
    combined_data = np.hstack([au_data, emotion_data])
    print(f"\n8. Combined data shape: {combined_data.shape}")
    print(f"   - Total features per frame: {combined_data.shape[1]} ({au_data.shape[1]} AUs + {emotion_data.shape[1]} emotions)")
    
    # Save outputs
    print("\n9. Saving outputs...")
    
    # Save full CSV with all columns
    csv_full_path = f"{output_prefix}_full_predictions.csv"
    video_prediction.to_csv(csv_full_path, index=False)
    print(f"   ✓ Full CSV saved: {csv_full_path}")
    
    # Save AU + Emotion CSV (compact)
    au_emotion_df = video_prediction[['frame', 'approx_time'] + au_columns + available_emotions].copy()
    csv_compact_path = f"{output_prefix}_au_emotions.csv"
    au_emotion_df.to_csv(csv_compact_path, index=False)
    print(f"   ✓ Compact CSV saved: {csv_compact_path}")
    
    # Save AU data as NPY
    au_npy_path = f"{output_prefix}_au_data.npy"
    np.save(au_npy_path, au_data)
    print(f"   ✓ AU data (NPY) saved: {au_npy_path}")
    print(f"     Shape: {au_data.shape}")
    
    # Save emotion data as NPY
    emotion_npy_path = f"{output_prefix}_emotion_data.npy"
    np.save(emotion_npy_path, emotion_data)
    print(f"   ✓ Emotion data (NPY) saved: {emotion_npy_path}")
    print(f"     Shape: {emotion_data.shape}")
    
    # Save combined AU + emotion data as NPY
    combined_npy_path = f"{output_prefix}_au_emotions.npy"
    np.save(combined_npy_path, combined_data)
    print(f"   ✓ Combined AU+Emotion data (NPY) saved: {combined_npy_path}")
    print(f"     Shape: {combined_data.shape}")
    
    # Save column names for reference
    column_names_path = f"{output_prefix}_column_names.txt"
    with open(column_names_path, 'w') as f:
        f.write("Action Units:\n")
        for i, col in enumerate(au_columns):
            f.write(f"  Column {i}: {col}\n")
        f.write(f"\nEmotions:\n")
        for i, col in enumerate(available_emotions):
            f.write(f"  Column {len(au_columns) + i}: {col}\n")
    print(f"   ✓ Column names saved: {column_names_path}")
    
    # Display summary statistics
    print("\n10. Summary Statistics:")
    print("\n    Action Units (mean values across all frames):")
    for col in au_columns[:10]:  # Show first 10 AUs
        mean_val = video_prediction[col].mean()
        std_val = video_prediction[col].std()
        print(f"      {col}: mean={mean_val:.4f}, std={std_val:.4f}")
    if len(au_columns) > 10:
        print(f"      ... and {len(au_columns) - 10} more AUs")
    
    print("\n    Emotions (mean values across all frames):")
    for col in available_emotions:
        mean_val = video_prediction[col].mean()
        std_val = video_prediction[col].std()
        print(f"      {col}: mean={mean_val:.4f}, std={std_val:.4f}")
    
    # Frame count summary
    actual_frames = len(video_prediction)
    print(f"\n11. Processing summary:")
    print(f"    - Total frames processed: {actual_frames}")
    print(f"    - Video duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    print(f"    - Average processing speed: {actual_frames / duration * skip_frames:.2f} fps")
    
    print("\n" + "=" * 70)
    print("✓ Extraction complete!")
    print("=" * 70)
    
    print("\nOutput files created:")
    print(f"  1. {csv_full_path} - Full predictions with all columns")
    print(f"  2. {csv_compact_path} - Compact CSV with frame, time, AUs, emotions")
    print(f"  3. {au_npy_path} - NumPy array of AU values only")
    print(f"  4. {emotion_npy_path} - NumPy array of emotion values only")
    print(f"  5. {combined_npy_path} - NumPy array of combined AU+emotion values")
    print(f"  6. {column_names_path} - Column names reference")
    
    print("\nData format:")
    print(f"  - AU array shape: (num_frames={au_data.shape[0]}, num_aus={au_data.shape[1]})")
    print(f"  - Emotion array shape: (num_frames={emotion_data.shape[0]}, num_emotions={emotion_data.shape[1]})")
    print(f"  - Combined array shape: (num_frames={combined_data.shape[0]}, num_features={combined_data.shape[1]})")
    
    return {
        'au_data': au_data,
        'emotion_data': emotion_data,
        'combined_data': combined_data,
        'au_columns': au_columns,
        'emotion_columns': available_emotions,
        'predictions': video_prediction,
        'num_frames': actual_frames
    }


def main():
    """Main function to parse arguments and run extraction"""
    parser = argparse.ArgumentParser(
        description='Extract Action Units and Emotions from video at 30fps',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a video with automatic output naming
  python extract_au_emotions.py input_video.mp4
  
  # Process with custom output prefix
  python extract_au_emotions.py input_video.mp4 --output my_results
  
  # Process at different fps
  python extract_au_emotions.py input_video.mp4 --fps 24
        """
    )
    
    parser.add_argument(
        'video_path',
        type=str,
        help='Path to input video file'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output prefix for generated files (default: video filename)'
    )
    
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Target frames per second for processing (default: 30)'
    )
    
    args = parser.parse_args()
    
    try:
        results = extract_au_emotions(
            video_path=args.video_path,
            output_prefix=args.output,
            target_fps=args.fps
        )
        
        print("\n" + "=" * 70)
        print("Success! You can now load the data in Python:")
        print("=" * 70)
        print("""
import numpy as np
import pandas as pd

# Load NPY files
au_data = np.load('OUTPUT_au_data.npy')
emotion_data = np.load('OUTPUT_emotion_data.npy')
combined_data = np.load('OUTPUT_au_emotions.npy')

# Load CSV file
df = pd.read_csv('OUTPUT_au_emotions.csv')

# Access data
print(f"AU data shape: {au_data.shape}")
print(f"Emotion data shape: {emotion_data.shape}")
print(f"First frame AUs: {au_data[0]}")
print(f"First frame emotions: {emotion_data[0]}")
        """)
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

