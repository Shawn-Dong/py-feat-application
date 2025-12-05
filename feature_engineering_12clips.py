#!/usr/bin/env python3
"""
Feature engineering for AU and emotion data (12 clips version).
Extract various statistical and temporal features for ML models.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.signal import find_peaks

def load_data():
    """Load the combined features."""
    csv_path = Path("/Users/dyst/py-feat/features_12clips/combined_au_features.csv")
    df = pd.read_csv(csv_path)
    return df

def extract_statistical_features(data, prefix=''):
    """Extract statistical features from a series of values."""
    features = {}
    features[f'{prefix}_mean'] = data.mean()
    features[f'{prefix}_median'] = data.median()
    features[f'{prefix}_std'] = data.std()
    features[f'{prefix}_min'] = data.min()
    features[f'{prefix}_max'] = data.max()
    features[f'{prefix}_range'] = data.max() - data.min()
    
    # Percentiles
    features[f'{prefix}_q25'] = data.quantile(0.25)
    features[f'{prefix}_q75'] = data.quantile(0.75)
    features[f'{prefix}_iqr'] = features[f'{prefix}_q75'] - features[f'{prefix}_q25']
    
    # Shape statistics
    features[f'{prefix}_skew'] = data.skew()
    features[f'{prefix}_kurtosis'] = data.kurtosis()
    
    return features

def extract_temporal_features(data, prefix=''):
    """Extract temporal/dynamic features."""
    features = {}
    
    # Rate of change (first derivative)
    diff = data.diff().dropna()
    features[f'{prefix}_mean_change'] = diff.mean()
    features[f'{prefix}_std_change'] = diff.std()
    features[f'{prefix}_abs_change'] = diff.abs().mean()
    
    # Acceleration (second derivative)
    diff2 = diff.diff().dropna()
    features[f'{prefix}_mean_accel'] = diff2.mean()
    features[f'{prefix}_std_accel'] = diff2.std()
    
    # Peak detection
    peaks, properties = find_peaks(data.values, prominence=0.1)
    features[f'{prefix}_num_peaks'] = len(peaks)
    features[f'{prefix}_peak_prominence_mean'] = properties['prominences'].mean() if len(peaks) > 0 else 0
    
    # Trend (linear regression slope)
    if len(data) > 1:
        x = np.arange(len(data))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, data)
        features[f'{prefix}_trend_slope'] = slope
        features[f'{prefix}_trend_r2'] = r_value ** 2
    else:
        features[f'{prefix}_trend_slope'] = 0
        features[f'{prefix}_trend_r2'] = 0
    
    return features

def extract_au_combinations(clip_df, au_cols):
    """Extract AU combination features."""
    features = {}
    
    # Upper face AUs (brows and eyes)
    upper_face = ['AU01', 'AU02', 'AU04', 'AU05', 'AU06', 'AU07']
    upper_face_present = [au for au in upper_face if au in au_cols]
    if upper_face_present:
        features['upper_face_mean'] = clip_df[upper_face_present].mean().mean()
        features['upper_face_std'] = clip_df[upper_face_present].std().mean()
        features['upper_face_max'] = clip_df[upper_face_present].max().max()
    
    # Lower face AUs (nose, mouth, chin)
    lower_face = ['AU09', 'AU10', 'AU11', 'AU12', 'AU14', 'AU15', 'AU17', 'AU20', 
                  'AU23', 'AU24', 'AU25', 'AU26', 'AU28']
    lower_face_present = [au for au in lower_face if au in au_cols]
    if lower_face_present:
        features['lower_face_mean'] = clip_df[lower_face_present].mean().mean()
        features['lower_face_std'] = clip_df[lower_face_present].std().mean()
        features['lower_face_max'] = clip_df[lower_face_present].max().max()
    
    # Number of highly activated AUs (> 0.5)
    features['num_high_aus'] = (clip_df[au_cols] > 0.5).sum(axis=1).mean()
    
    # AU diversity (how many different AUs are active)
    features['au_diversity'] = (clip_df[au_cols] > 0.3).sum(axis=1).mean()
    
    return features

def extract_emotion_features(clip_df, emotion_cols):
    """Extract emotion-specific features."""
    features = {}
    
    # Dominant emotion frequency
    dominant_emotions = clip_df[emotion_cols].idxmax(axis=1)
    for emotion in emotion_cols:
        features[f'freq_{emotion}_dominant'] = (dominant_emotions == emotion).mean()
    
    # Emotion intensity (max probability across all emotions)
    features['max_emotion_intensity'] = clip_df[emotion_cols].max(axis=1).mean()
    
    # Emotion variability (entropy-like measure)
    emotion_variance = clip_df[emotion_cols].std(axis=1).mean()
    features['emotion_variability'] = emotion_variance
    
    # Binary: is emotionally expressive? (not just neutral)
    features['expressiveness'] = (clip_df['neutral'] < 0.5).mean()
    
    return features

def extract_clip_level_features(df):
    """Extract features at the clip level (aggregating frames)."""
    au_cols = [col for col in df.columns if col.startswith('AU')]
    emotion_cols = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']
    
    clip_features_list = []
    
    for sample_id in df['sample_id'].unique():
        clip_df = df[df['sample_id'] == sample_id]
        
        features = {
            'sample_id': sample_id,
            'video_id': clip_df['video_id'].iloc[0],
            'clip_id': clip_df['clip_id'].iloc[0],
            'num_frames': len(clip_df)
        }
        
        # Extract features for each AU
        for au in au_cols:
            stat_features = extract_statistical_features(clip_df[au], prefix=au)
            features.update(stat_features)
            
            temp_features = extract_temporal_features(clip_df[au], prefix=au)
            features.update(temp_features)
        
        # Extract features for each emotion
        for emotion in emotion_cols:
            stat_features = extract_statistical_features(clip_df[emotion], prefix=emotion)
            features.update(stat_features)
            
            temp_features = extract_temporal_features(clip_df[emotion], prefix=emotion)
            features.update(temp_features)
        
        # AU combination features
        au_combo_features = extract_au_combinations(clip_df, au_cols)
        features.update(au_combo_features)
        
        # Emotion-specific features
        emotion_features = extract_emotion_features(clip_df, emotion_cols)
        features.update(emotion_features)
        
        clip_features_list.append(features)
    
    return pd.DataFrame(clip_features_list)

def extract_simple_aggregations(df):
    """Extract simple aggregated features (mean, std, min, max per clip)."""
    au_cols = [col for col in df.columns if col.startswith('AU')]
    emotion_cols = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']
    all_feature_cols = au_cols + emotion_cols
    
    # Aggregate by clip
    agg_dict = {}
    for col in all_feature_cols:
        agg_dict[col] = ['mean', 'std', 'min', 'max', 'median']
    
    clip_features = df.groupby('sample_id').agg(agg_dict)
    
    # Flatten column names
    clip_features.columns = ['_'.join(col).strip() for col in clip_features.columns.values]
    clip_features = clip_features.reset_index()
    
    # Add metadata
    metadata = df.groupby('sample_id')[['video_id', 'clip_id']].first().reset_index()
    clip_features = clip_features.merge(metadata, on='sample_id')
    
    return clip_features

def main():
    print("\nLoading data (12 clips version)...")
    df = load_data()
    
    print(f"Loaded {len(df)} frames from {df['video_id'].nunique()} videos")
    print(f"Total clips: {df['sample_id'].nunique()}")
    
    # Extract simple aggregations
    print("\n" + "="*80)
    print("EXTRACTING FEATURES (12 clips per video)...")
    print("="*80)
    
    output_folder = Path("/Users/dyst/py-feat/features_12clips")
    
    print("\n[1/2] Extracting simple aggregated features (FAST)...")
    simple_features = extract_simple_aggregations(df)
    output_file = output_folder / "clip_features_simple.csv"
    simple_features.to_csv(output_file, index=False)
    print(f"✓ Saved: {output_file}")
    print(f"  Shape: {simple_features.shape[0]} clips × {simple_features.shape[1]} features")
    
    print("\n[2/2] Extracting full engineered features (COMPREHENSIVE)...")
    full_features = extract_clip_level_features(df)
    output_file = output_folder / "clip_features_full.csv"
    full_features.to_csv(output_file, index=False)
    print(f"✓ Saved: {output_file}")
    print(f"  Shape: {full_features.shape[0]} clips × {full_features.shape[1]} features")
    
    print("\n" + "="*80)
    print("✓ Feature extraction complete!")
    print("="*80)
    print("\nGenerated files in features_12clips/:")
    print("  1. combined_au_features.csv   - Frame-level data (27 features)")
    print("  2. clip_features_simple.csv   - Simple aggregations (138 features)")
    print("  3. clip_features_full.csv     - Full engineered features (~562 features)")
    print()
    print(f"Data summary:")
    print(f"  - Videos: {df['video_id'].nunique()}")
    print(f"  - Clips per video: 12")
    print(f"  - Total clips: {df['sample_id'].nunique()}")
    print(f"  - Total frames: {len(df)}")
    print(f"  - Avg frames per clip: {len(df) / df['sample_id'].nunique():.1f}")

if __name__ == "__main__":
    main()

