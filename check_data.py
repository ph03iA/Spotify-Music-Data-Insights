#!/usr/bin/env python3
"""
Check Spotify Data Structure
"""

try:
    import pandas as pd
    
    # Load the data
    print("📊 Loading Spotify dataset...")
    df = pd.read_csv('data/archive/data.csv')
    
    print(f"\n📈 Dataset Information:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    print(f"\n📊 Sample Data:")
    print(df.head())
    
    print(f"\n🎵 Data Types:")
    print(df.dtypes)
    
    print(f"\n🔢 Missing Values:")
    print(df.isnull().sum())
    
    print(f"\n📈 Basic Statistics:")
    print(df.describe())
    
    # Check for specific columns we need
    required_columns = ['popularity', 'danceability', 'energy', 'tempo', 'genre']
    available_columns = [col for col in required_columns if col in df.columns]
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    print(f"\n✅ Available columns for analysis: {available_columns}")
    if missing_columns:
        print(f"⚠️  Missing columns: {missing_columns}")
    
except ImportError as e:
    print(f"❌ Error: {e}")
    print("Please install pandas: pip install pandas")
except Exception as e:
    print(f"❌ Error reading data: {e}") 