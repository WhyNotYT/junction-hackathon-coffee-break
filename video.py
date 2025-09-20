import os
import numpy as np
import pandas as pd
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

@dataclass
class FilmFrame:
    """Represents a column as a 'film frame' for machine vision processing"""
    column_name: str
    data: np.ndarray
    anomaly_scores: np.ndarray
    image_representation: np.ndarray
    statistics: Dict

class CSVAnomalyDetector:
    def __init__(self, folder_path: str, chunk_size: int = 50000):
        self.folder_path = Path(folder_path)
        self.chunk_size = chunk_size
        self.films: Dict[str, List[FilmFrame]] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        
    def load_csv_efficiently(self, file_path: Path) -> pd.DataFrame:
        """Load CSV in chunks and return processed dataframe"""
        print(f"Processing {file_path.name}...")
        
        # First pass: get column info and data types
        sample_df = pd.read_csv(file_path, nrows=1000)
        numeric_columns = sample_df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_columns:
            print(f"No numeric columns found in {file_path.name}")
            return None
            
        # Read only numeric columns in chunks
        chunks = []
        for chunk in pd.read_csv(file_path, chunksize=self.chunk_size, usecols=numeric_columns):
            # Basic cleaning
            chunk = chunk.dropna()
            chunk = chunk.replace([np.inf, -np.inf], np.nan).dropna()
            if not chunk.empty:
                chunks.append(chunk)
        
        if chunks:
            return pd.concat(chunks, ignore_index=True)
        return None
    
    def column_to_film_frame(self, column_data: pd.Series, column_name: str) -> FilmFrame:
        """Convert a column into a film frame representation"""
        data = column_data.values.astype(np.float32)
        
        # Remove outliers using IQR method for better visualization
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Calculate anomaly scores using Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42, n_jobs=-1)
        anomaly_scores = iso_forest.fit_predict(data.reshape(-1, 1))
        anomaly_scores = (anomaly_scores == -1).astype(float)  # Convert to 0/1
        
        # Create image representation (film frame)
        # Reshape data into a 2D grid for visualization
        target_size = int(np.sqrt(len(data)) * 1.2)  # Slightly rectangular
        if target_size * target_size < len(data):
            target_size += 1
            
        # Pad data to fit perfect square/rectangle
        padded_length = target_size * target_size
        padded_data = np.pad(data, (0, padded_length - len(data)), mode='constant', constant_values=np.nan)
        padded_anomalies = np.pad(anomaly_scores, (0, padded_length - len(anomaly_scores)), mode='constant', constant_values=0)
        
        # Reshape to 2D grid
        data_grid = padded_data.reshape(target_size, target_size)
        anomaly_grid = padded_anomalies.reshape(target_size, target_size)
        
        # Normalize for visualization
        data_normalized = np.nan_to_num(data_grid)
        if data_normalized.max() != data_normalized.min():
            data_normalized = (data_normalized - data_normalized.min()) / (data_normalized.max() - data_normalized.min())
        
        # Create RGB image: data as intensity, anomalies as red channel
        image_rep = np.zeros((target_size, target_size, 3), dtype=np.float32)
        image_rep[:, :, 0] = anomaly_grid * 255  # Red for anomalies
        image_rep[:, :, 1] = data_normalized * 255  # Green for data intensity
        image_rep[:, :, 2] = data_normalized * 255  # Blue for data intensity
        
        # Apply morphological operations for better anomaly detection
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        anomaly_enhanced = cv2.morphologyEx(anomaly_grid.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        image_rep[:, :, 0] = anomaly_enhanced * 255
        
        # Calculate statistics
        stats = {
            'mean': float(np.nanmean(data)),
            'std': float(np.nanstd(data)),
            'min': float(np.nanmin(data)),
            'max': float(np.nanmax(data)),
            'anomaly_count': int(np.sum(anomaly_scores)),
            'anomaly_percentage': float(np.sum(anomaly_scores) / len(anomaly_scores) * 100),
            'skewness': float(pd.Series(data).skew()),
            'kurtosis': float(pd.Series(data).kurtosis())
        }
        
        return FilmFrame(
            column_name=column_name,
            data=data,
            anomaly_scores=anomaly_scores,
            image_representation=image_rep,
            statistics=stats
        )
    
    def process_folder(self, max_workers: int = 4):
        """Process all CSV files in the folder"""
        csv_files = list(self.folder_path.glob("*.csv"))
        
        if not csv_files:
            print(f"No CSV files found in {self.folder_path}")
            return
        
        print(f"Found {len(csv_files)} CSV files to process...")
        
        # Process files in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for csv_file in csv_files:
                future = executor.submit(self._process_single_file, csv_file)
                futures.append((csv_file.stem, future))
            
            for filename, future in futures:
                try:
                    films = future.result(timeout=300)  # 5 minute timeout per file
                    if films:
                        self.films[filename] = films
                        print(f"✓ Processed {filename}: {len(films)} columns")
                except Exception as e:
                    print(f"✗ Error processing {filename}: {str(e)}")
    
    def _process_single_file(self, csv_file: Path) -> List[FilmFrame]:
        """Process a single CSV file"""
        df = self.load_csv_efficiently(csv_file)
        if df is None or df.empty:
            return []
        
        films = []
        for column in df.columns:
            try:
                film_frame = self.column_to_film_frame(df[column], f"{csv_file.stem}_{column}")
                films.append(film_frame)
            except Exception as e:
                print(f"Error processing column {column} in {csv_file.name}: {str(e)}")
                continue
        
        return films
    
    def create_anomaly_dashboard(self, output_dir: str = "anomaly_visualizations"):
        """Create comprehensive visualization dashboard"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        if not self.films:
            print("No films to visualize. Run process_folder() first.")
            return
        
        # Create film strip visualization
        self._create_film_strip_visualization(output_path)
        
        # Create individual anomaly heatmaps
        self._create_anomaly_heatmaps(output_path)
        
        # Create statistical summary
        self._create_statistical_summary(output_path)
        
        print(f"Visualizations saved to {output_path}")
    
    def _create_film_strip_visualization(self, output_path: Path):
        """Create a film strip showing all columns as frames"""
        all_films = []
        for file_films in self.films.values():
            all_films.extend(file_films)
        
        if not all_films:
            return
            
        # Sort by anomaly percentage for better visualization
        all_films.sort(key=lambda f: f.statistics['anomaly_percentage'], reverse=True)
        
        # Create film strip (top 20 most anomalous)
        n_films = min(20, len(all_films))
        cols = 5
        rows = (n_films + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows))
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
        
        for i, film in enumerate(all_films[:n_films]):
            ax = axes[i]
            ax.imshow(film.image_representation.astype(np.uint8))
            ax.set_title(f"{film.column_name}\nAnomalies: {film.statistics['anomaly_percentage']:.1f}%", 
                        fontsize=8)
            ax.axis('off')
        
        # Hide unused subplots
        for i in range(n_films, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path / "film_strip_anomalies.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_anomaly_heatmaps(self, output_path: Path):
        """Create detailed heatmaps for top anomalous columns"""
        all_films = []
        for file_films in self.films.values():
            all_films.extend(file_films)
        
        # Sort by anomaly percentage
        all_films.sort(key=lambda f: f.statistics['anomaly_percentage'], reverse=True)
        
        # Create detailed heatmaps for top 5 anomalous columns
        for i, film in enumerate(all_films[:5]):
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original data heatmap
            data_grid = film.image_representation[:, :, 1]  # Green channel (data)
            im1 = ax1.imshow(data_grid, cmap='viridis')
            ax1.set_title(f"Data Distribution\n{film.column_name}")
            plt.colorbar(im1, ax=ax1)
            
            # Anomaly heatmap
            anomaly_grid = film.image_representation[:, :, 0]  # Red channel (anomalies)
            im2 = ax2.imshow(anomaly_grid, cmap='Reds')
            ax2.set_title(f"Anomaly Locations\n{film.statistics['anomaly_count']} anomalies")
            plt.colorbar(im2, ax=ax2)
            
            # Combined view
            im3 = ax3.imshow(film.image_representation.astype(np.uint8))
            ax3.set_title("Combined View\n(Red=Anomalies)")
            
            plt.tight_layout()
            plt.savefig(output_path / f"anomaly_detail_{i+1}_{film.column_name.replace('/', '_')}.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_statistical_summary(self, output_path: Path):
        """Create statistical summary of all films"""
        all_films = []
        for file_films in self.films.values():
            all_films.extend(file_films)
        
        if not all_films:
            return
        
        # Collect statistics
        stats_data = []
        for film in all_films:
            stats = film.statistics.copy()
            stats['column_name'] = film.column_name
            stats_data.append(stats)
        
        stats_df = pd.DataFrame(stats_data)
        
        # Create summary plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Anomaly percentage distribution
        axes[0, 0].hist(stats_df['anomaly_percentage'], bins=20, alpha=0.7, color='red')
        axes[0, 0].set_title('Anomaly Percentage Distribution')
        axes[0, 0].set_xlabel('Anomaly Percentage')
        axes[0, 0].set_ylabel('Count')
        
        # Top anomalous columns
        top_anomalous = stats_df.nlargest(10, 'anomaly_percentage')
        axes[0, 1].barh(range(len(top_anomalous)), top_anomalous['anomaly_percentage'])
        axes[0, 1].set_yticks(range(len(top_anomalous)))
        axes[0, 1].set_yticklabels([name.split('_')[-1][:15] for name in top_anomalous['column_name']], fontsize=8)
        axes[0, 1].set_title('Top 10 Most Anomalous Columns')
        axes[0, 1].set_xlabel('Anomaly Percentage')
        
        # Data distribution characteristics
        axes[0, 2].scatter(stats_df['skewness'], stats_df['kurtosis'], 
                          c=stats_df['anomaly_percentage'], cmap='Reds', alpha=0.6)
        axes[0, 2].set_title('Data Distribution Characteristics')
        axes[0, 2].set_xlabel('Skewness')
        axes[0, 2].set_ylabel('Kurtosis')
        
        # Anomaly count vs data range
        data_range = stats_df['max'] - stats_df['min']
        axes[1, 0].scatter(data_range, stats_df['anomaly_count'], alpha=0.6)
        axes[1, 0].set_title('Anomaly Count vs Data Range')
        axes[1, 0].set_xlabel('Data Range')
        axes[1, 0].set_ylabel('Anomaly Count')
        axes[1, 0].set_xscale('log')
        
        # Standard deviation vs anomalies
        axes[1, 1].scatter(stats_df['std'], stats_df['anomaly_percentage'], alpha=0.6)
        axes[1, 1].set_title('Standard Deviation vs Anomaly Percentage')
        axes[1, 1].set_xlabel('Standard Deviation')
        axes[1, 1].set_ylabel('Anomaly Percentage')
        axes[1, 1].set_xscale('log')
        
        # Summary statistics table (top 10 anomalous)
        axes[1, 2].axis('tight')
        axes[1, 2].axis('off')
        summary_table = top_anomalous[['column_name', 'anomaly_percentage', 'anomaly_count']].head(10)
        summary_table['column_name'] = summary_table['column_name'].str.split('_').str[-1].str[:20]
        table = axes[1, 2].table(cellText=summary_table.values,
                                colLabels=['Column', 'Anomaly %', 'Count'],
                                cellLoc='center',
                                loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        axes[1, 2].set_title('Top Anomalous Columns Summary')
        
        plt.tight_layout()
        plt.savefig(output_path / "statistical_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save detailed statistics to CSV
        stats_df.to_csv(output_path / "detailed_statistics.csv", index=False)
    
    def get_anomaly_summary(self) -> Dict:
        """Get summary of anomaly detection results"""
        if not self.films:
            return {"error": "No films processed"}
        
        all_films = []
        for file_films in self.films.values():
            all_films.extend(file_films)
        
        total_anomalies = sum(film.statistics['anomaly_count'] for film in all_films)
        total_data_points = sum(len(film.data) for film in all_films)
        
        most_anomalous = max(all_films, key=lambda f: f.statistics['anomaly_percentage'])
        
        return {
            "total_files_processed": len(self.films),
            "total_columns_processed": len(all_films),
            "total_data_points": total_data_points,
            "total_anomalies": total_anomalies,
            "overall_anomaly_percentage": (total_anomalies / total_data_points) * 100 if total_data_points > 0 else 0,
            "most_anomalous_column": {
                "name": most_anomalous.column_name,
                "anomaly_percentage": most_anomalous.statistics['anomaly_percentage']
            }
        }

# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = CSVAnomalyDetector(folder_path="output", chunk_size=50000)
    
    # Process all CSV files in the folder
    detector.process_folder(max_workers=4)
    
    # Create visualizations
    detector.create_anomaly_dashboard()
    
    # Get summary
    summary = detector.get_anomaly_summary()
    print("\n=== ANOMALY DETECTION SUMMARY ===")
    for key, value in summary.items():
        print(f"{key}: {value}")