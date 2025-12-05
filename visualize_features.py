#!/usr/bin/env python3
"""
Visualize the extracted AU features and emotions across videos and clips.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_features():
    """Load the combined feature CSV."""
    feature_file = Path("/Users/dyst/py-feat/features/combined_au_features.csv")
    df = pd.read_csv(feature_file)
    print(f"Loaded {len(df)} frames from {df['video_id'].nunique()} videos")
    return df

def plot_emotion_distribution(df, output_dir):
    """Plot emotion distribution across all videos and clips."""
    emotion_cols = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Emotion Distribution Analysis', fontsize=16, fontweight='bold')
    
    # 1. Average emotion across all videos
    ax1 = axes[0, 0]
    emotion_means = df[emotion_cols].mean()
    colors = ['red', 'green', 'purple', 'gold', 'blue', 'orange', 'gray']
    ax1.bar(emotion_cols, emotion_means, color=colors, alpha=0.7)
    ax1.set_title('Average Emotion Intensity (All Videos)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mean Probability')
    ax1.set_ylim(0, 1)
    ax1.grid(axis='y', alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 2. Emotion by video
    ax2 = axes[0, 1]
    emotion_by_video = df.groupby('video_id')[emotion_cols].mean()
    emotion_by_video.plot(kind='bar', ax=ax2, width=0.8)
    ax2.set_title('Average Emotions by Video', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Mean Probability')
    ax2.set_xlabel('Video ID')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax2.grid(axis='y', alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 3. Emotion distribution (box plot)
    ax3 = axes[1, 0]
    df[emotion_cols].boxplot(ax=ax3, patch_artist=True)
    ax3.set_title('Emotion Probability Distribution', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Probability')
    ax3.grid(axis='y', alpha=0.3)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 4. Dominant emotion distribution
    ax4 = axes[1, 1]
    df['dominant_emotion'] = df[emotion_cols].idxmax(axis=1)
    emotion_counts = df['dominant_emotion'].value_counts()
    colors_dict = dict(zip(emotion_cols, colors))
    emotion_colors = [colors_dict[e] for e in emotion_counts.index]
    ax4.pie(emotion_counts, labels=emotion_counts.index, autopct='%1.1f%%', 
            colors=emotion_colors, startangle=90)
    ax4.set_title('Dominant Emotion Distribution', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    output_file = output_dir / 'emotion_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved emotion analysis: {output_file}")
    plt.close()

def plot_au_heatmap(df, output_dir):
    """Plot AU activation heatmap."""
    au_cols = [col for col in df.columns if col.startswith('AU')]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Action Unit (AU) Analysis', fontsize=16, fontweight='bold')
    
    # 1. Average AU activation by video
    ax1 = axes[0]
    au_by_video = df.groupby('video_id')[au_cols].mean()
    sns.heatmap(au_by_video.T, cmap='YlOrRd', cbar_kws={'label': 'Mean Activation'},
                ax=ax1, vmin=0, vmax=1)
    ax1.set_title('Average AU Activation by Video', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Video ID')
    ax1.set_ylabel('Action Unit')
    
    # 2. AU correlation matrix
    ax2 = axes[1]
    au_corr = df[au_cols].corr()
    sns.heatmap(au_corr, cmap='coolwarm', center=0, ax=ax2, 
                cbar_kws={'label': 'Correlation'}, vmin=-1, vmax=1)
    ax2.set_title('AU Correlation Matrix', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    output_file = output_dir / 'au_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved AU analysis: {output_file}")
    plt.close()

def plot_temporal_patterns(df, output_dir):
    """Plot temporal patterns of emotions across clips."""
    emotion_cols = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']
    
    # Average emotion by clip position (1-6)
    emotion_by_clip = df.groupby('clip_id')[emotion_cols].mean()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for emotion in emotion_cols:
        ax.plot(emotion_by_clip.index, emotion_by_clip[emotion], 
                marker='o', linewidth=2, label=emotion)
    
    ax.set_title('Emotion Patterns Across Clip Position (All Videos)', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Clip Number (1-6)', fontsize=12)
    ax.set_ylabel('Mean Probability', fontsize=12)
    ax.set_xticks([1, 2, 3, 4, 5, 6])
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / 'temporal_patterns.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved temporal patterns: {output_file}")
    plt.close()

def plot_video_comparison(df, output_dir):
    """Create a detailed comparison across all videos."""
    emotion_cols = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']
    videos = sorted(df['video_id'].unique())
    
    fig, axes = plt.subplots(len(videos), 1, figsize=(14, 3*len(videos)))
    fig.suptitle('Emotion Trajectories by Video and Clip', fontsize=16, fontweight='bold')
    
    for idx, video in enumerate(videos):
        ax = axes[idx] if len(videos) > 1 else axes
        video_df = df[df['video_id'] == video]
        
        # Calculate mean emotion for each clip
        for clip_id in range(1, 7):
            clip_df = video_df[video_df['clip_id'] == clip_id]
            if len(clip_df) > 0:
                clip_emotions = clip_df[emotion_cols].mean()
                x_pos = clip_id - 0.5
                ax.bar([x_pos + i*0.1 for i in range(len(emotion_cols))], 
                       clip_emotions, width=0.1, 
                       label=emotion_cols if clip_id == 1 else None)
        
        ax.set_title(f'{video}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Clip Number')
        ax.set_ylabel('Mean Probability')
        ax.set_xlim(0.5, 6.5)
        ax.set_xticks([1, 2, 3, 4, 5, 6])
        ax.grid(axis='y', alpha=0.3)
        
        if idx == 0:
            ax.legend(emotion_cols, loc='upper right', ncol=7, fontsize=8)
    
    plt.tight_layout()
    output_file = output_dir / 'video_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved video comparison: {output_file}")
    plt.close()

def generate_statistics_report(df, output_dir):
    """Generate a detailed statistics report."""
    emotion_cols = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']
    au_cols = [col for col in df.columns if col.startswith('AU')]
    
    report_file = output_dir / 'statistics_report.txt'
    
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("FEATURE EXTRACTION STATISTICS REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Basic stats
        f.write("DATASET OVERVIEW\n")
        f.write("-"*80 + "\n")
        f.write(f"Total frames: {len(df):,}\n")
        f.write(f"Total videos: {df['video_id'].nunique()}\n")
        f.write(f"Total clips: {df['sample_id'].nunique()}\n")
        f.write(f"Average frames per video: {len(df) / df['video_id'].nunique():.1f}\n")
        f.write(f"Average frames per clip: {len(df) / df['sample_id'].nunique():.1f}\n\n")
        
        # Emotion statistics
        f.write("EMOTION STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write("Mean emotion probabilities:\n")
        for emotion in emotion_cols:
            mean_val = df[emotion].mean()
            std_val = df[emotion].std()
            min_val = df[emotion].min()
            max_val = df[emotion].max()
            f.write(f"  {emotion:12s}: mean={mean_val:.4f}, std={std_val:.4f}, "
                   f"min={min_val:.4f}, max={max_val:.4f}\n")
        
        f.write("\nDominant emotion counts:\n")
        df['dominant_emotion'] = df[emotion_cols].idxmax(axis=1)
        dominant_counts = df['dominant_emotion'].value_counts()
        for emotion, count in dominant_counts.items():
            percentage = (count / len(df)) * 100
            f.write(f"  {emotion:12s}: {count:6,} frames ({percentage:5.2f}%)\n")
        
        # AU statistics
        f.write("\n\nACTION UNIT STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write("Mean AU activation:\n")
        au_means = df[au_cols].mean().sort_values(ascending=False)
        for au, mean_val in au_means.items():
            f.write(f"  {au}: {mean_val:.4f}\n")
        
        # Per-video statistics
        f.write("\n\nPER-VIDEO STATISTICS\n")
        f.write("-"*80 + "\n")
        for video in sorted(df['video_id'].unique()):
            video_df = df[df['video_id'] == video]
            f.write(f"\n{video}:\n")
            f.write(f"  Total frames: {len(video_df)}\n")
            f.write(f"  Dominant emotion: {video_df['dominant_emotion'].mode()[0]}\n")
            f.write(f"  Mean emotions:\n")
            for emotion in emotion_cols:
                f.write(f"    {emotion:12s}: {video_df[emotion].mean():.4f}\n")
    
    print(f"✓ Saved statistics report: {report_file}")

def main():
    # Create output directory
    output_dir = Path("/Users/dyst/py-feat/features/visualizations")
    output_dir.mkdir(exist_ok=True)
    
    # Load features
    print("\nLoading features...")
    df = load_features()
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_emotion_distribution(df, output_dir)
    plot_au_heatmap(df, output_dir)
    plot_temporal_patterns(df, output_dir)
    plot_video_comparison(df, output_dir)
    
    # Generate statistics report
    print("\nGenerating statistics report...")
    generate_statistics_report(df, output_dir)
    
    print("\n" + "="*80)
    print("✓ All visualizations and reports generated successfully!")
    print(f"  Output directory: {output_dir}")
    print("="*80)

if __name__ == "__main__":
    main()

