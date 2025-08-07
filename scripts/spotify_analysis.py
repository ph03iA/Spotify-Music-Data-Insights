#!/usr/bin/env python3
"""
Spotify Music Data Insights Analysis
====================================

This script performs comprehensive analysis of Spotify music data including:
- Popularity analysis by genre
- Tempo distribution analysis
- Danceability vs Energy correlation
- Audio feature correlations
- Creative insights and visualizations

Author: Prajwal
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os
from pathlib import Path

# Set up styling
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

# Create visualizations directory if it doesn't exist
Path("../visualizations").mkdir(exist_ok=True)

class SpotifyAnalyzer:
    """Main class for Spotify data analysis"""
    
    def __init__(self, data_path=None):
        """Initialize the analyzer with data path"""
        self.data_path = data_path or "../data/archive/data.csv"
        self.df = None
        self.audio_features = [
            'danceability', 'energy', 'key', 'loudness', 'mode',
            'speechiness', 'acousticness', 'instrumentalness',
            'liveness', 'valence', 'tempo', 'duration_ms'
        ]
        
    def load_data(self, data_path=None):
        """Load Spotify dataset"""
        if data_path:
            self.data_path = data_path
            
        # Try to load the dataset
        try:
            if os.path.exists(self.data_path):
                print(f"📊 Loading real Spotify dataset from: {self.data_path}")
                self.df = pd.read_csv(self.data_path)
                
                # Clean and prepare the data
                self._prepare_data()
                
                print(f"✅ Real data loaded successfully! Shape: {self.df.shape}")
                return True
            else:
                print(f"❌ Data file not found: {self.data_path}")
                print("📊 Creating sample data for demonstration...")
                self._create_sample_data()
                return False
                
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            print("📊 Creating sample data for demonstration...")
            self._create_sample_data()
            return False
    
    def _prepare_data(self):
        """Prepare the real Spotify data for analysis"""
        print("🔧 Preparing data for analysis...")
        
        # Check available columns
        print(f"Available columns: {list(self.df.columns)}")
        
        # Handle missing columns by creating them if needed
        if 'genre' not in self.df.columns:
            # Try to create genre from other available columns
            if 'explicit' in self.df.columns:
                self.df['genre'] = 'unknown'  # Default genre
            else:
                self.df['genre'] = 'unknown'
        
        # Ensure we have the required columns for analysis
        required_columns = ['popularity', 'danceability', 'energy', 'tempo']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        
        if missing_columns:
            print(f"⚠️  Missing columns: {missing_columns}")
            print("Creating sample data for missing features...")
            self._create_sample_data()
            return
        
        # Clean the data
        # Remove rows with missing values in key columns
        self.df = self.df.dropna(subset=['popularity', 'danceability', 'energy', 'tempo'])
        
        # Ensure data types are correct
        numeric_columns = ['popularity', 'danceability', 'energy', 'tempo', 'loudness', 'valence']
        for col in numeric_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Remove outliers (optional)
        self.df = self.df[self.df['popularity'] >= 0]
        self.df = self.df[self.df['popularity'] <= 100]
        
        print(f"✅ Data prepared! Final shape: {self.df.shape}")
    
    def _create_sample_data(self):
        """Create sample Spotify data for demonstration"""
        print("📊 Creating sample Spotify dataset for demonstration...")
        
        np.random.seed(42)
        n_samples = 10000
        
        # Generate realistic Spotify data
        genres = ['pop', 'hip-hop', 'rock', 'electronic', 'r&b', 'country', 'jazz', 'classical']
        
        data = {
            'track_name': [f"Track_{i}" for i in range(n_samples)],
            'artist_name': [f"Artist_{i % 100}" for i in range(n_samples)],
            'genre': np.random.choice(genres, n_samples, p=[0.3, 0.25, 0.2, 0.1, 0.08, 0.04, 0.02, 0.01]),
            'popularity': np.random.normal(50, 20, n_samples).clip(0, 100),
            'danceability': np.random.beta(2, 2, n_samples),
            'energy': np.random.beta(2, 2, n_samples),
            'key': np.random.randint(0, 12, n_samples),
            'loudness': np.random.normal(-10, 5, n_samples),
            'mode': np.random.choice([0, 1], n_samples),
            'speechiness': np.random.beta(1, 5, n_samples),
            'acousticness': np.random.beta(1, 3, n_samples),
            'instrumentalness': np.random.beta(1, 5, n_samples),
            'liveness': np.random.beta(1, 5, n_samples),
            'valence': np.random.beta(2, 2, n_samples),
            'tempo': np.random.normal(120, 30, n_samples).clip(50, 200),
            'duration_ms': np.random.normal(180000, 60000, n_samples).clip(30000, 600000)
        }
        
        self.df = pd.DataFrame(data)
        
        # Add some realistic correlations
        self.df['energy'] = self.df['energy'] + 0.3 * self.df['loudness'] / 10
        self.df['danceability'] = self.df['danceability'] + 0.2 * self.df['energy']
        self.df['popularity'] = self.df['popularity'] + 10 * self.df['danceability']
        
        # Clip values to valid ranges
        for col in ['energy', 'danceability', 'popularity']:
            self.df[col] = self.df[col].clip(0, 1 if col != 'popularity' else 100)
    
    def explore_data(self):
        """Perform initial data exploration"""
        print("\n🔍 EXPLORATORY DATA ANALYSIS")
        print("=" * 50)
        
        print(f"Dataset shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        print(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        print("\n📊 Basic Statistics:")
        print(self.df.describe())
        
        if 'genre' in self.df.columns:
            print("\n🎵 Genre Distribution:")
            genre_counts = self.df['genre'].value_counts()
            print(genre_counts)
        
        print("\n🔢 Missing Values:")
        print(self.df.isnull().sum())
        
        # Show unique values for categorical columns
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            print("\n📝 Categorical Columns:")
            for col in categorical_cols:
                print(f"{col}: {self.df[col].nunique()} unique values")
    
    def analyze_popularity_by_genre(self):
        """Analyze popularity distribution by genre"""
        print("\n📈 POPULARITY ANALYSIS BY GENRE")
        print("=" * 50)
        
        if 'genre' not in self.df.columns:
            print("⚠️  No genre column found. Skipping genre analysis.")
            return None
        
        # Calculate popularity statistics by genre
        genre_popularity = self.df.groupby('genre')['popularity'].agg([
            'count', 'mean', 'median', 'std', 'min', 'max'
        ]).round(2)
        
        print("Popularity Statistics by Genre:")
        print(genre_popularity)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Box plot
        plt.subplot(2, 2, 1)
        sns.boxplot(data=self.df, x='genre', y='popularity')
        plt.title('Popularity Distribution by Genre')
        plt.xticks(rotation=45)
        
        # Bar plot of mean popularity
        plt.subplot(2, 2, 2)
        genre_means = self.df.groupby('genre')['popularity'].mean().sort_values(ascending=False)
        genre_means.plot(kind='bar', color='skyblue')
        plt.title('Average Popularity by Genre')
        plt.xticks(rotation=45)
        
        # Popularity distribution
        plt.subplot(2, 2, 3)
        sns.histplot(data=self.df, x='popularity', bins=30, kde=True)
        plt.title('Overall Popularity Distribution')
        
        # Popularity vs genre count
        plt.subplot(2, 2, 4)
        genre_counts = self.df['genre'].value_counts()
        genre_pop_means = self.df.groupby('genre')['popularity'].mean()
        plt.scatter(genre_counts, genre_pop_means, s=100, alpha=0.7)
        for i, genre in enumerate(genre_counts.index):
            plt.annotate(genre, (genre_counts.iloc[i], genre_pop_means.iloc[i]), 
                        xytext=(5, 5), textcoords='offset points')
        plt.xlabel('Number of Tracks')
        plt.ylabel('Average Popularity')
        plt.title('Genre Count vs Average Popularity')
        
        plt.tight_layout()
        plt.savefig('../visualizations/popularity_by_genre.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return genre_popularity
    
    def analyze_tempo_distribution(self):
        """Analyze tempo distribution and patterns"""
        print("\n🎵 TEMPO DISTRIBUTION ANALYSIS")
        print("=" * 50)
        
        # Tempo statistics
        tempo_stats = self.df['tempo'].describe()
        print("Tempo Statistics:")
        print(tempo_stats)
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Overall tempo distribution
        axes[0, 0].hist(self.df['tempo'], bins=50, alpha=0.7, color='lightcoral')
        axes[0, 0].axvline(self.df['tempo'].mean(), color='red', linestyle='--', label=f'Mean: {self.df["tempo"].mean():.1f}')
        axes[0, 0].set_title('Overall Tempo Distribution')
        axes[0, 0].set_xlabel('Tempo (BPM)')
        axes[0, 0].legend()
        
        # Tempo by genre (if available)
        if 'genre' in self.df.columns:
            sns.boxplot(data=self.df, x='genre', y='tempo', ax=axes[0, 1])
            axes[0, 1].set_title('Tempo Distribution by Genre')
            axes[0, 1].tick_params(axis='x', rotation=45)
        else:
            axes[0, 1].text(0.5, 0.5, 'No genre data available', ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Tempo Distribution by Genre')
        
        # Tempo vs popularity
        axes[1, 0].scatter(self.df['tempo'], self.df['popularity'], alpha=0.5, s=20)
        axes[1, 0].set_xlabel('Tempo (BPM)')
        axes[1, 0].set_ylabel('Popularity')
        axes[1, 0].set_title('Tempo vs Popularity')
        
        # Popularity distribution by tempo ranges
        self.df['tempo_range'] = pd.cut(self.df['tempo'], bins=5, labels=['Very Slow', 'Slow', 'Medium', 'Fast', 'Very Fast'])
        sns.boxplot(data=self.df, x='tempo_range', y='popularity', ax=axes[1, 1])
        axes[1, 1].set_title('Popularity by Tempo Range')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('../visualizations/tempo_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Key insights
        print(f"\n🎯 Key Tempo Insights:")
        print(f"- Average tempo: {self.df['tempo'].mean():.1f} BPM")
        print(f"- Most common tempo range: {self.df['tempo_range'].mode().iloc[0]}")
        print(f"- Tempo range: {self.df['tempo'].min():.1f} - {self.df['tempo'].max():.1f} BPM")
        
        return tempo_stats
    
    def analyze_danceability_vs_energy(self):
        """Analyze correlation between danceability and energy"""
        print("\n💃 DANCEABILITY VS ENERGY ANALYSIS")
        print("=" * 50)
        
        # Calculate correlation
        correlation = self.df['danceability'].corr(self.df['energy'])
        print(f"Correlation coefficient: {correlation:.3f}")
        
        # Create visualizations
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Scatter plot
        axes[0, 0].scatter(self.df['danceability'], self.df['energy'], alpha=0.6, s=20)
        axes[0, 0].set_xlabel('Danceability')
        axes[0, 0].set_ylabel('Energy')
        axes[0, 0].set_title(f'Danceability vs Energy\nCorrelation: {correlation:.3f}')
        
        # Add trend line
        z = np.polyfit(self.df['danceability'], self.df['energy'], 1)
        p = np.poly1d(z)
        axes[0, 0].plot(self.df['danceability'], p(self.df['danceability']), "r--", alpha=0.8)
        
        # Distribution plots
        axes[0, 1].hist(self.df['danceability'], bins=30, alpha=0.7, color='lightblue', label='Danceability')
        axes[0, 1].set_title('Danceability Distribution')
        axes[0, 1].legend()
        
        axes[0, 2].hist(self.df['energy'], bins=30, alpha=0.7, color='lightcoral', label='Energy')
        axes[0, 2].set_title('Energy Distribution')
        axes[0, 2].legend()
        
        # Joint plot
        sns.kdeplot(data=self.df, x='danceability', y='energy', ax=axes[1, 0], cmap='viridis')
        axes[1, 0].set_title('Density Plot: Danceability vs Energy')
        
        # Popularity analysis
        axes[1, 1].scatter(self.df['danceability'], self.df['popularity'], alpha=0.6, s=20, color='green')
        axes[1, 1].set_xlabel('Danceability')
        axes[1, 1].set_ylabel('Popularity')
        axes[1, 1].set_title('Danceability vs Popularity')
        
        axes[1, 2].scatter(self.df['energy'], self.df['popularity'], alpha=0.6, s=20, color='orange')
        axes[1, 2].set_xlabel('Energy')
        axes[1, 2].set_ylabel('Popularity')
        axes[1, 2].set_title('Energy vs Popularity')
        
        plt.tight_layout()
        plt.savefig('../visualizations/danceability_energy_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create interactive plot
        fig = px.scatter(self.df, x='danceability', y='energy', 
                        color='popularity', size='popularity',
                        hover_data=['track_name', 'artist_name', 'genre'] if 'track_name' in self.df.columns else None,
                        title=f'Interactive: Danceability vs Energy (Correlation: {correlation:.3f})')
        fig.write_html('../visualizations/danceability_energy_interactive.html')
        
        return correlation
    
    def create_correlation_heatmap(self):
        """Create correlation heatmap for all audio features"""
        print("\n🔥 AUDIO FEATURE CORRELATION HEATMAP")
        print("=" * 50)
        
        # Select audio features that exist in the dataset
        available_features = [feat for feat in self.audio_features if feat in self.df.columns]
        features_for_corr = available_features + ['popularity']
        
        if len(features_for_corr) < 2:
            print("⚠️  Not enough features available for correlation analysis")
            return None
        
        correlation_matrix = self.df[features_for_corr].corr()
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, 
                   mask=mask,
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   square=True,
                   fmt='.2f',
                   cbar_kws={'shrink': 0.8})
        plt.title('Audio Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig('../visualizations/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Find strongest correlations
        print("\n🔍 Strongest Correlations:")
        correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                correlations.append((correlation_matrix.columns[i], 
                                  correlation_matrix.columns[j], 
                                  corr_val))
        
        # Sort by absolute correlation value
        correlations.sort(key=lambda x: abs(x[2]), reverse=True)
        
        print("Top 10 Strongest Correlations:")
        for i, (feat1, feat2, corr) in enumerate(correlations[:10]):
            print(f"{i+1:2d}. {feat1:15s} ↔ {feat2:15s}: {corr:6.3f}")
        
        return correlation_matrix
    
    def generate_creative_insights(self):
        """Generate creative insights from the data"""
        print("\n💡 CREATIVE INSIGHTS")
        print("=" * 50)
        
        insights = []
        
        # 1. Genre popularity insights (if available)
        if 'genre' in self.df.columns:
            genre_pop = self.df.groupby('genre')['popularity'].mean().sort_values(ascending=False)
            top_genre = genre_pop.index[0]
            insights.append(f"🎵 {top_genre.title()} is the most popular genre with average popularity of {genre_pop.iloc[0]:.1f}")
        
        # 2. Tempo insights
        popular_tracks = self.df[self.df['popularity'] > 70]
        if len(popular_tracks) > 0:
            avg_popular_tempo = popular_tracks['tempo'].mean()
            insights.append(f"🎶 Most popular songs have an average tempo of {avg_popular_tempo:.1f} BPM")
        
        # 3. Danceability insights
        dance_corr = self.df['danceability'].corr(self.df['popularity'])
        insights.append(f"💃 Danceability has a {dance_corr:.3f} correlation with popularity")
        
        # 4. Energy insights
        energy_corr = self.df['energy'].corr(self.df['popularity'])
        insights.append(f"⚡ Energy has a {energy_corr:.3f} correlation with popularity")
        
        # 5. Feature combination insights
        self.df['dance_energy_score'] = (self.df['danceability'] + self.df['energy']) / 2
        dance_energy_corr = self.df['dance_energy_score'].corr(self.df['popularity'])
        insights.append(f"🎯 Combined danceability-energy score has {dance_energy_corr:.3f} correlation with popularity")
        
        # 6. Genre-specific insights (if available)
        if 'genre' in self.df.columns:
            for genre in self.df['genre'].unique():
                genre_data = self.df[self.df['genre'] == genre]
                if len(genre_data) > 50:  # Only for genres with sufficient data
                    avg_tempo = genre_data['tempo'].mean()
                    avg_dance = genre_data['danceability'].mean()
                    insights.append(f"🎸 {genre.title()}: Avg tempo {avg_tempo:.1f} BPM, danceability {avg_dance:.2f}")
        
        # Print insights
        for i, insight in enumerate(insights, 1):
            print(f"{i:2d}. {insight}")
        
        return insights
    
    def create_interactive_dashboard(self):
        """Create an interactive dashboard with Plotly"""
        print("\n📊 Creating Interactive Dashboard...")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Popularity by Genre', 'Tempo Distribution', 
                          'Danceability vs Energy', 'Feature Correlations'),
            specs=[[{"type": "bar"}, {"type": "histogram"}],
                   [{"type": "scatter"}, {"type": "heatmap"}]]
        )
        
        # 1. Popularity by genre (if available)
        if 'genre' in self.df.columns:
            genre_pop = self.df.groupby('genre')['popularity'].mean().sort_values(ascending=False)
            fig.add_trace(
                go.Bar(x=genre_pop.index, y=genre_pop.values, name="Avg Popularity"),
                row=1, col=1
            )
        else:
            # Show overall popularity distribution instead
            fig.add_trace(
                go.Histogram(x=self.df['popularity'], nbinsx=30, name="Popularity Distribution"),
                row=1, col=1
            )
        
        # 2. Tempo distribution
        fig.add_trace(
            go.Histogram(x=self.df['tempo'], nbinsx=30, name="Tempo Distribution"),
            row=1, col=2
        )
        
        # 3. Danceability vs Energy
        fig.add_trace(
            go.Scatter(x=self.df['danceability'], y=self.df['energy'], 
                      mode='markers', name="Danceability vs Energy",
                      marker=dict(size=5, opacity=0.6)),
            row=2, col=1
        )
        
        # 4. Correlation heatmap
        available_features = [feat for feat in self.audio_features if feat in self.df.columns]
        features_for_corr = available_features + ['popularity']
        
        if len(features_for_corr) >= 2:
            corr_matrix = self.df[features_for_corr].corr()
            
            fig.add_trace(
                go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns,
                          colorscale='RdBu', zmid=0),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="Spotify Music Data Insights Dashboard",
            height=800,
            showlegend=False
        )
        
        fig.write_html('../visualizations/interactive_dashboard.html')
        print("✅ Interactive dashboard saved to visualizations/interactive_dashboard.html")
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("🎵 SPOTIFY MUSIC DATA INSIGHTS ANALYSIS")
        print("=" * 60)
        
        # Load data
        self.load_data()
        
        # Run all analyses
        self.explore_data()
        self.analyze_popularity_by_genre()
        self.analyze_tempo_distribution()
        self.analyze_danceability_vs_energy()
        self.create_correlation_heatmap()
        self.generate_creative_insights()
        self.create_interactive_dashboard()
        
        print("\n✅ Analysis complete! Check the visualizations/ folder for all charts.")
        print("📊 Key files generated:")
        print("   - popularity_by_genre.png")
        print("   - tempo_analysis.png")
        print("   - danceability_energy_analysis.png")
        print("   - correlation_heatmap.png")
        print("   - danceability_energy_interactive.html")
        print("   - interactive_dashboard.html")

if __name__ == "__main__":
    # Initialize analyzer
    analyzer = SpotifyAnalyzer()
    
    # Run complete analysis
    analyzer.run_complete_analysis() 