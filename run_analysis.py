#!/usr/bin/env python3
"""
Spotify Music Data Insights - Main Runner
=========================================

This script runs the complete Spotify analysis pipeline including:
- Data loading/downloading
- Comprehensive analysis
- Visualization generation
- Interactive dashboard creation

Author: Data Analyst
Date: 2024
"""

import sys
import os
from pathlib import Path

def main():
    """Main function to run the complete analysis"""
    print("🎵 SPOTIFY MUSIC DATA INSIGHTS")
    print("=" * 50)
    print("Starting comprehensive analysis with REAL Spotify data...")
    
    # Check if real data exists
    data_path = Path("data/archive/data.csv")
    if data_path.exists():
        print(f"✅ Found real Spotify dataset: {data_path}")
        print(f"📊 File size: {data_path.stat().st_size / (1024*1024):.1f} MB")
    else:
        print("⚠️  Real data not found, will use sample data")
    
    # Add scripts directory to path
    scripts_dir = Path("scripts")
    if scripts_dir.exists():
        sys.path.insert(0, str(scripts_dir))
    
    try:
        # Import and run the analysis
        from scripts.spotify_analysis import SpotifyAnalyzer

        
        # Initialize analyzer with real data path
        analyzer = SpotifyAnalyzer(data_path=str(data_path) if data_path.exists() else None)
        
        # Run complete analysis
        analyzer.run_complete_analysis()
        
        print("\n🎉 Analysis completed successfully!")
        print("\n📁 Generated files:")
        print("   - visualizations/popularity_by_genre.png")
        print("   - visualizations/tempo_analysis.png")
        print("   - visualizations/danceability_energy_analysis.png")
        print("   - visualizations/correlation_heatmap.png")
        print("   - visualizations/danceability_energy_interactive.html")
        print("   - visualizations/interactive_dashboard.html")
        
        print("\n🚀 Next steps:")
        print("1. Open the HTML files in your browser for interactive visualizations")
        print("2. Run: jupyter notebook notebooks/spotify_analysis.ipynb")
        print("3. Explore the data further with your own analysis!")
        
        # Show some quick stats about the analysis
        if hasattr(analyzer, 'df') and analyzer.df is not None:
            print(f"\n📈 Analysis Summary:")
            print(f"   - Total tracks analyzed: {len(analyzer.df):,}")
            print(f"   - Average popularity: {analyzer.df['popularity'].mean():.1f}")
            print(f"   - Average tempo: {analyzer.df['tempo'].mean():.1f} BPM")
            print(f"   - Average danceability: {analyzer.df['danceability'].mean():.3f}")
            print(f"   - Average energy: {analyzer.df['energy'].mean():.3f}")
        
    except ImportError as e:
        print(f"❌ Error importing analysis module: {e}")
        print("Make sure you're in the correct directory and all files are present.")
        
    except Exception as e:
        print(f"❌ Error running analysis: {e}")
        print("Please check the error message above and try again.")

if __name__ == "__main__":
    main() 