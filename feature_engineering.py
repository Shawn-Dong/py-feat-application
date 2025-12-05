#!/usr/bin/env python3
"""
Feature engineering for AU and emotion data.
Extract various statistical and temporal features for ML models.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.signal import find_peaks

def load_data():
    """Load the combined features."""
    csv_path = Path("/Users/dyst/py-feat/features/combined_au_features.csv")
    df = pd.read_csv(csv_path)
    return df

def extract_statistical_features(data, prefix=''):
    """
    Extract statistical features from a series of values.
    
    Features:
    - Mean, Median, Std, Min, Max
    - Percentiles (25, 50, 75)
    - Range, IQR
    - Skewness, Kurtosis
    """
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
    """
    Extract temporal/dynamic features.
    
    Features:
    - First derivative (rate of change)
    - Acceleration (second derivative)
    - Number of peaks
    - Peak properties
    """
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
    """
    Extract AU combination features.
    Upper face AUs, Lower face AUs, co-occurrences.
    """
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
    """
    Extract emotion-specific features.
    """
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
    """
    Extract features at the clip level (aggregating frames).
    """
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
    """
    Extract simple aggregated features (mean, std, min, max per clip).
    This is faster and good for initial modeling.
    """
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

def create_feature_summary():
    """Create a summary of all possible features."""
    au_cols = [f'AU{i:02d}' for i in [1, 2, 4, 5, 6, 7, 9, 10, 11, 12, 14, 15, 17, 20, 23, 24, 25, 26, 28, 43]]
    emotion_cols = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']
    
    feature_types = {
        'Statistical Features (per feature)': [
            'mean', 'median', 'std', 'min', 'max', 'range',
            'q25 (25th percentile)', 'q75 (75th percentile)', 'iqr',
            'skew', 'kurtosis'
        ],
        'Temporal Features (per feature)': [
            'mean_change (velocity)', 'std_change', 'abs_change',
            'mean_accel (acceleration)', 'std_accel',
            'num_peaks', 'peak_prominence_mean',
            'trend_slope', 'trend_r2'
        ],
        'AU Combination Features': [
            'upper_face_mean', 'upper_face_std', 'upper_face_max',
            'lower_face_mean', 'lower_face_std', 'lower_face_max',
            'num_high_aus (>0.5)', 'au_diversity (>0.3)'
        ],
        'Emotion-Specific Features': [
            'freq_[emotion]_dominant (7 emotions)',
            'max_emotion_intensity',
            'emotion_variability',
            'expressiveness'
        ]
    }
    
    total_features = 0
    
    print("="*80)
    print("FEATURE EXTRACTION SUMMARY")
    print("="*80)
    print()
    
    # Raw features
    print("1. RAW FEATURES (Frame-level)")
    print("-"*80)
    print(f"   - Action Units: {len(au_cols)} features")
    print(f"   - Emotions: {len(emotion_cols)} features")
    print(f"   Total raw features: {len(au_cols) + len(emotion_cols)}")
    print()
    
    # Aggregated features
    print("2. AGGREGATED FEATURES (Clip-level)")
    print("-"*80)
    
    base_features = len(au_cols) + len(emotion_cols)  # 27
    
    for feature_type, features in feature_types.items():
        print(f"\n   {feature_type}:")
        for feat in features:
            print(f"      - {feat}")
        
        if feature_type == 'Statistical Features (per feature)':
            count = base_features * 11  # 11 statistical features per base feature
            print(f"   Subtotal: {base_features} features × 11 stats = {count} features")
            total_features += count
        
        elif feature_type == 'Temporal Features (per feature)':
            count = base_features * 8  # 8 temporal features per base feature
            print(f"   Subtotal: {base_features} features × 8 temporal = {count} features")
            total_features += count
        
        elif feature_type == 'AU Combination Features':
            count = 8
            print(f"   Subtotal: {count} features")
            total_features += count
        
        elif feature_type == 'Emotion-Specific Features':
            count = 7 + 3  # 7 frequency features + 3 others
            print(f"   Subtotal: {count} features")
            total_features += count
    
    print()
    print("="*80)
    print(f"TOTAL ENGINEERED FEATURES: {total_features}")
    print("="*80)
    print()
    
    print("3. FEATURE EXTRACTION OPTIONS")
    print("-"*80)
    print("   Option A: Simple Aggregations (Fast)")
    print(f"      - {base_features} features × 5 aggregations (mean, std, min, max, median)")
    print(f"      - Total: {base_features * 5} features")
    print()
    print("   Option B: Full Feature Engineering (Comprehensive)")
    print(f"      - All statistical, temporal, and combination features")
    print(f"      - Total: {total_features} features")
    print()
    print("   Option C: Raw Frame-level Features")
    print(f"      - Use frame-by-frame data directly")
    print(f"      - Total: {base_features} features per frame")
    print(f"      - Good for: LSTMs, RNNs, Transformers")
    print()

def main():
    print("\nLoading data...")
    df = load_data()
    
    print("Creating feature extraction summary...")
    create_feature_summary()
    
    # Extract simple aggregations
    print("\n" + "="*80)
    print("EXTRACTING FEATURES...")
    print("="*80)
    
    print("\n[1/2] Extracting simple aggregated features (FAST)...")
    simple_features = extract_simple_aggregations(df)
    output_file = Path("/Users/dyst/py-feat/features/clip_features_simple.csv")
    simple_features.to_csv(output_file, index=False)
    print(f"✓ Saved: {output_file}")
    print(f"  Shape: {simple_features.shape[0]} clips × {simple_features.shape[1]} features")
    
    print("\n[2/2] Extracting full engineered features (COMPREHENSIVE)...")
    full_features = extract_clip_level_features(df)
    output_file = Path("/Users/dyst/py-feat/features/clip_features_full.csv")
    full_features.to_csv(output_file, index=False)
    print(f"✓ Saved: {output_file}")
    print(f"  Shape: {full_features.shape[0]} clips × {full_features.shape[1]} features")
    
    print("\n" + "="*80)
    print("✓ Feature extraction complete!")
    print("="*80)
    print("\nGenerated files:")
    print("  1. clip_features_simple.csv   - Simple aggregations (135 features)")
    print("  2. clip_features_full.csv     - Full engineered features (~500+ features)")
    print("  3. combined_au_features.csv   - Original frame-level data (27 features)")

if __name__ == "__main__":
    main()

