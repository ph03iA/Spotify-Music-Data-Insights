#!/usr/bin/env python3
"""
Spotify Data Download Script
============================

This script downloads the Spotify dataset from Kaggle or creates sample data
if the download fails. It handles both real and demonstration scenarios.

Author: Data Analyst
Date: 2024
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import requests
import zipfile
import warnings

warnings.filterwarnings('ignore')

def create_sample_spotify_data():
    """Create realistic sample Spotify data for demonstration"""
    print("📊 Creating sample Spotify dataset...")
    
    np.random.seed(42)
    n_samples = 15000
    
    # Generate realistic Spotify data with proper distributions
    genres = ['pop', 'hip-hop', 'rock', 'electronic', 'r&b', 'country', 'jazz', 'classical']
    genre_weights = [0.35, 0.25, 0.20, 0.10, 0.05, 0.03, 0.01, 0.01]
    
    # Create base data
    data = {
        'track_name': [f"Track_{i:05d}" for i in range(n_samples)],
        'artist_name': [f"Artist_{i % 200}" for i in range(n_samples)],
        'genre': np.random.choice(genres, n_samples, p=genre_weights),
        'popularity': np.random.normal(50, 20, n_samples).clip(0, 100),
        'danceability': np.random.beta(2, 2, n_samples),
        'energy': np.random.beta(2, 2, n_samples),
        'key': np.random.randint(0, 12, n_samples),
        'loudness': np.random.normal(-10, 5, n_samples),
        'mode': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
        'speechiness': np.random.beta(1, 5, n_samples),
        'acousticness': np.random.beta(1, 3, n_samples),
        'instrumentalness': np.random.beta(1, 5, n_samples),
        'liveness': np.random.beta(1, 5, n_samples),
        'valence': np.random.beta(2, 2, n_samples),
        'tempo': np.random.normal(120, 30, n_samples).clip(50, 200),
        'duration_ms': np.random.normal(180000, 60000, n_samples).clip(30000, 600000)
    }
    
    df = pd.DataFrame(data)
    
    # Add realistic correlations and genre-specific patterns
    # Energy and loudness correlation
    df['energy'] = df['energy'] + 0.3 * (df['loudness'] + 10) / 20
    df['energy'] = df['energy'].clip(0, 1)
    
    # Danceability influenced by energy
    df['danceability'] = df['danceability'] + 0.2 * df['energy']
    df['danceability'] = df['danceability'].clip(0, 1)
    
    # Popularity influenced by danceability and energy
    df['popularity'] = df['popularity'] + 15 * df['danceability'] + 10 * df['energy']
    df['popularity'] = df['popularity'].clip(0, 100)
    
    # Genre-specific adjustments
    genre_adjustments = {
        'pop': {'tempo': 10, 'danceability': 0.1, 'energy': 0.1},
        'hip-hop': {'tempo': -5, 'danceability': 0.15, 'speechiness': 0.2},
        'rock': {'energy': 0.2, 'loudness': 2, 'acousticness': -0.1},
        'electronic': {'tempo': 15, 'danceability': 0.2, 'energy': 0.15},
        'r&b': {'tempo': -10, 'danceability': 0.05, 'valence': 0.1},
        'country': {'acousticness': 0.3, 'tempo': -15, 'valence': 0.15},
        'jazz': {'instrumentalness': 0.4, 'acousticness': 0.2, 'tempo': -20},
        'classical': {'instrumentalness': 0.8, 'acousticness': 0.4, 'tempo': -30}
    }
    
    for genre, adjustments in genre_adjustments.items():
        mask = df['genre'] == genre
        for feature, adjustment in adjustments.items():
            if feature in df.columns:
                df.loc[mask, feature] = (df.loc[mask, feature] + adjustment).clip(0, 1)
    
    # Add some realistic track names and artists
    popular_artists = [
        'Taylor Swift', 'Drake', 'The Weeknd', 'Ed Sheeran', 'Ariana Grande',
        'Post Malone', 'Billie Eilish', 'Dua Lipa', 'Justin Bieber', 'BTS',
        'Bad Bunny', 'Olivia Rodrigo', 'Doja Cat', 'Lil Baby', 'Travis Scott'
    ]
    
    # Replace some artist names with popular ones
    for i in range(min(len(popular_artists), 100)):
        df.loc[i * 150, 'artist_name'] = popular_artists[i % len(popular_artists)]
        df.loc[i * 150, 'track_name'] = f"Hit Song {i+1}"
        df.loc[i * 150, 'popularity'] = np.random.randint(80, 100)
    
    return df

def download_kaggle_dataset():
    """Attempt to download Spotify dataset from Kaggle"""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        
        # Download Spotify dataset
        print("📥 Downloading Spotify dataset from Kaggle...")
        api.dataset_download_files('maharshipandya/-spotify-tracks-dataset', path='../data', unzip=True)
        
        # Look for the CSV file
        data_dir = Path('../data')
        csv_files = list(data_dir.glob('*.csv'))
        
        if csv_files:
            print(f"✅ Dataset downloaded successfully: {csv_files[0]}")
            return str(csv_files[0])
        else:
            print("❌ No CSV file found in downloaded data")
            return None
            
    except Exception as e:
        print(f"❌ Error downloading from Kaggle: {e}")
        print("📝 Note: You need to set up Kaggle API credentials to download real data")
        return None

def download_sample_dataset():
    """Download a sample dataset from a public source"""
    try:
        # Try to download a sample dataset from GitHub
        url = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-01-21/spotify_songs.csv"
        
        print("📥 Downloading sample Spotify dataset...")
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            # Save the data
            data_path = Path('../data/spotify_songs.csv')
            data_path.parent.mkdir(exist_ok=True)
            
            with open(data_path, 'wb') as f:
                f.write(response.content)
            
            print(f"✅ Sample dataset downloaded: {data_path}")
            return str(data_path)
        else:
            print(f"❌ Failed to download sample dataset: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Error downloading sample dataset: {e}")
        return None

def main():
    """Main function to download or create Spotify data"""
    print("🎵 SPOTIFY DATA DOWNLOAD")
    print("=" * 40)
    
    # Create data directory
    data_dir = Path('../data')
    data_dir.mkdir(exist_ok=True)
    
    # Try to download real data first
    data_path = download_kaggle_dataset()
    
    if not data_path:
        # Try alternative download
        data_path = download_sample_dataset()
    
    if not data_path:
        # Create sample data
        print("📊 Creating sample dataset for demonstration...")
        df = create_sample_spotify_data()
        
        # Save sample data
        data_path = data_dir / 'spotify_sample_data.csv'
        df.to_csv(data_path, index=False)
        print(f"✅ Sample dataset created: {data_path}")
        print(f"📊 Dataset shape: {df.shape}")
        print(f"🎵 Genres: {df['genre'].nunique()}")
        print(f"🎤 Artists: {df['artist_name'].nunique()}")
        
        # Show sample statistics
        print("\n📈 Sample Statistics:")
        print(f"- Average popularity: {df['popularity'].mean():.1f}")
        print(f"- Average tempo: {df['tempo'].mean():.1f} BPM")
        print(f"- Average danceability: {df['danceability'].mean():.3f}")
        print(f"- Average energy: {df['energy'].mean():.3f}")
        
        # Show genre distribution
        print("\n🎵 Genre Distribution:")
        genre_counts = df['genre'].value_counts()
        for genre, count in genre_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {genre:12s}: {count:5d} tracks ({percentage:5.1f}%)")
    
    else:
        print(f"✅ Real dataset available: {data_path}")
        print("📊 You can now run the analysis with real Spotify data!")
    
    print("\n🚀 Next steps:")
    print("1. Run: python scripts/spotify_analysis.py")
    print("2. Check the visualizations/ folder for charts")
    print("3. Explore with Jupyter: jupyter notebook notebooks/")

if __name__ == "__main__":
    main() 