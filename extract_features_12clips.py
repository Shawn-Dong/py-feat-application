#!/usr/bin/env python3
"""
Extract and combine AU emotion features from multiple videos.
Each 6-minute video is divided into 12 30-second clips.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import glob

def process_video_file(csv_path, video_id, num_clips=12):
    """
    Process a single video CSV file and divide it into clips.
    
    Parameters:
    -----------
    csv_path : str
        Path to the CSV file
    video_id : str
        Video identifier (e.g., AIMB3084)
    num_clips : int
        Number of clips to divide the video into (default: 12)
    
    Returns:
    --------
    pd.DataFrame
        Processed dataframe with video_id and clip_id columns
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Calculate total number of frames
    total_frames = len(df)
    
    # Divide into equal clips
    frames_per_clip = total_frames / num_clips
    
    # Add video_id column
    df['video_id'] = video_id
    
    # Add clip_id column (1-12)
    clip_ids = (df.index / frames_per_clip).astype(int) + 1
    df['clip_id'] = np.minimum(clip_ids, num_clips)
    
    # Add a combined identifier
    df['sample_id'] = df['video_id'] + '_clip' + df['clip_id'].astype(str)
    
    print(f"Processed {video_id}: {total_frames} frames -> {num_clips} clips")
    print(f"  Frames per clip: ~{frames_per_clip:.1f}")
    
    return df

def main():
    # Set paths
    au_folder = Path("/Users/dyst/py-feat/AU")
    output_folder = Path("/Users/dyst/py-feat/features_12clips")
    output_folder.mkdir(exist_ok=True)
    
    # Find all au_emotions.csv files (excluding full_predictions)
    pattern = str(au_folder / "*_au_emotions.csv")
    csv_files = sorted(glob.glob(pattern))
    
    print(f"Found {len(csv_files)} CSV files to process:\n")
    
    # Process each file
    all_data = []
    video_ids = []
    
    for csv_file in csv_files:
        # Extract video ID from filename
        filename = Path(csv_file).name
        video_id = filename.replace('_rotated_au_emotions.csv', '')
        video_ids.append(video_id)
        
        print(f"Processing: {filename}")
        df = process_video_file(csv_file, video_id, num_clips=12)
        all_data.append(df)
        print()
    
    # Combine all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Reorder columns to put identifiers first
    id_cols = ['sample_id', 'video_id', 'clip_id', 'frame', 'approx_time']
    other_cols = [col for col in combined_df.columns if col not in id_cols]
    combined_df = combined_df[id_cols + other_cols]
    
    # Save the combined features
    output_file = output_folder / "combined_au_features.csv"
    combined_df.to_csv(output_file, index=False)
    
    print("="*70)
    print(f"✓ Successfully combined all features!")
    print(f"  Total videos: {len(video_ids)}")
    print(f"  Total clips: {len(video_ids) * 12}")
    print(f"  Total frames: {len(combined_df)}")
    print(f"  Output file: {output_file}")
    print()
    
    # Print summary statistics
    print("Summary by video:")
    print(combined_df.groupby('video_id').size())
    print()
    
    print("Clips per video:")
    clips_per_video = combined_df.groupby('video_id')['clip_id'].nunique()
    print(clips_per_video)
    print()
    
    print("Sample of frames per clip:")
    clip_summary = combined_df.groupby(['video_id', 'clip_id']).size().reset_index(name='frames')
    print(clip_summary.head(20).to_string(index=False))
    print("...")
    print()
    
    # Save summary statistics
    summary_file = output_folder / "feature_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("AU Feature Extraction Summary (12 clips per video)\n")
        f.write("="*70 + "\n\n")
        f.write(f"Total videos: {len(video_ids)}\n")
        f.write(f"Total clips: {len(video_ids) * 12}\n")
        f.write(f"Total frames: {len(combined_df)}\n\n")
        f.write("Video IDs:\n")
        for vid in video_ids:
            f.write(f"  - {vid}\n")
        f.write("\n")
        f.write("Columns in combined feature file:\n")
        for col in combined_df.columns:
            f.write(f"  - {col}\n")
        f.write("\n")
        f.write("Frames per video:\n")
        f.write(str(combined_df.groupby('video_id').size()))
        f.write("\n\n")
        f.write("Frames per clip (first 24 clips):\n")
        f.write(clip_summary.head(24).to_string(index=False))
    
    print(f"✓ Summary saved to: {summary_file}")
    
    # Print feature column names
    print("\nFeature columns (excluding identifiers):")
    feature_cols = [col for col in combined_df.columns if col not in id_cols]
    print(f"  Total: {len(feature_cols)} features")
    print(f"  AU features: {len([c for c in feature_cols if c.startswith('AU')])}")
    print(f"  Emotion features: {len([c for c in feature_cols if not c.startswith('AU')])}")

if __name__ == "__main__":
    main()

