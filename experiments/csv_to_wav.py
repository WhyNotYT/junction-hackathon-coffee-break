import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import soundfile as sf
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

class CSVMusicAnomalyDetector:
    def __init__(self, folder_path, sample_rate=44100, contamination=0.1):
        """
        Initialize the CSV to Music Anomaly Detector
        
        Args:
            folder_path: Path to folder containing CSV files
            sample_rate: Audio sample rate (default 44100 Hz)
            contamination: Expected proportion of outliers (default 0.1 = 10%)
        """
        self.folder_path = folder_path
        self.sample_rate = sample_rate
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.isolation_forest = None
        self.data = None
        self.anomaly_scores = None
        self.is_anomaly = None
        
        # Musical parameters
        self.base_frequencies = [
            261.63,  # C4
            293.66,  # D4
            329.63,  # E4
            349.23,  # F4
            392.00,  # G4
            440.00,  # A4
            493.88,  # B4
            523.25,  # C5
        ]
        
    def load_csv_files(self):
        """Load and combine all CSV files from the folder"""
        print("Loading CSV files...")
        all_dataframes = []
        
        csv_files = [f for f in os.listdir(self.folder_path) if f.endswith('.csv')]
        
        if not csv_files:
            raise ValueError(f"No CSV files found in {self.folder_path}")
            
        for filename in csv_files:
            file_path = os.path.join(self.folder_path, filename)
            try:
                # Use chunking for large files
                chunk_list = []
                chunk_size = 10000
                
                for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                    # Select only numeric columns
                    numeric_chunk = chunk.select_dtypes(include=[np.number])
                    if not numeric_chunk.empty:
                        chunk_list.append(numeric_chunk)
                
                if chunk_list:
                    df = pd.concat(chunk_list, ignore_index=True)
                    all_dataframes.append(df)
                    print(f"Loaded {filename}: {len(df)} rows, {len(df.columns)} numeric columns")
                    
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue
        
        if not all_dataframes:
            raise ValueError("No valid numeric data found in CSV files")
            
        # Combine all dataframes
        self.data = pd.concat(all_dataframes, ignore_index=True)
        
        # Handle missing values
        self.data = self.data.fillna(self.data.median())
        
        print(f"Combined dataset: {len(self.data)} rows, {len(self.data.columns)} columns")
        return self.data
    
    def detect_anomalies(self):
        """Use Isolation Forest to detect anomalies"""
        print("Detecting anomalies...")
        
        if self.data is None:
            raise ValueError("No data loaded. Call load_csv_files() first.")
        
        # Sample data if too large (for efficiency)
        if len(self.data) > 100000:
            sample_data = self.data.sample(n=100000, random_state=42)
            print(f"Sampling {len(sample_data)} rows for anomaly detection")
        else:
            sample_data = self.data
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(sample_data)
        
        # Fit Isolation Forest
        self.isolation_forest = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_jobs=-1
        )
        
        sample_predictions = self.isolation_forest.fit_predict(scaled_data)
        sample_scores = self.isolation_forest.score_samples(scaled_data)
        
        # Apply to full dataset in chunks
        print("Applying anomaly detection to full dataset...")
        chunk_size = 10000
        all_predictions = []
        all_scores = []
        
        for i in range(0, len(self.data), chunk_size):
            chunk = self.data.iloc[i:i+chunk_size]
            scaled_chunk = self.scaler.transform(chunk)
            
            chunk_predictions = self.isolation_forest.predict(scaled_chunk)
            chunk_scores = self.isolation_forest.score_samples(scaled_chunk)
            
            all_predictions.extend(chunk_predictions)
            all_scores.extend(chunk_scores)
        
        self.is_anomaly = np.array(all_predictions) == -1
        self.anomaly_scores = np.array(all_scores)
        
        anomaly_count = np.sum(self.is_anomaly)
        print(f"Found {anomaly_count} anomalies ({anomaly_count/len(self.data)*100:.2f}%)")
        
        return self.is_anomaly, self.anomaly_scores
    
    def data_to_frequencies(self, data_subset, duration_per_row=0.1):
        """Convert data rows to frequencies for each 'instrument' (column)"""
        # Normalize each column to frequency ranges
        frequencies = []
        
        for col_idx, column in enumerate(data_subset.columns):
            col_data = data_subset[column].values
            
            # Normalize to 0-1 range
            col_min, col_max = col_data.min(), col_data.max()
            if col_max > col_min:
                normalized = (col_data - col_min) / (col_max - col_min)
            else:
                normalized = np.ones_like(col_data) * 0.5
            
            # Map to frequency range (2 octaves above base frequency)
            base_freq = self.base_frequencies[col_idx % len(self.base_frequencies)]
            freq_range = base_freq * 3  # Up to 3x the base frequency
            col_frequencies = base_freq + normalized * freq_range
            
            frequencies.append(col_frequencies)
        
        return np.array(frequencies)
    
    def generate_audio(self, frequencies, duration_per_row=0.1, fade_duration=0.01):
        """Generate audio from frequency data"""
        total_duration = len(frequencies[0]) * duration_per_row
        total_samples = int(total_duration * self.sample_rate)
        
        audio = np.zeros(total_samples)
        samples_per_row = int(duration_per_row * self.sample_rate)
        fade_samples = int(fade_duration * self.sample_rate)
        
        # Create fade in/out envelope
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        
        for row_idx in range(len(frequencies[0])):
            start_sample = row_idx * samples_per_row
            end_sample = min(start_sample + samples_per_row, total_samples)
            row_duration = (end_sample - start_sample) / self.sample_rate
            
            if row_duration <= 0:
                break
            
            # Generate time array for this row
            t = np.linspace(0, row_duration, end_sample - start_sample)
            row_audio = np.zeros_like(t)
            
            # Add each 'instrument' (column)
            for instrument_idx, freq_series in enumerate(frequencies):
                freq = freq_series[row_idx]
                # Use different waveforms for variety
                if instrument_idx % 3 == 0:
                    wave = np.sin(2 * np.pi * freq * t)  # Sine wave
                elif instrument_idx % 3 == 1:
                    wave = signal.square(2 * np.pi * freq * t) * 0.3  # Square wave (quieter)
                else:
                    wave = signal.sawtooth(2 * np.pi * freq * t) * 0.3  # Sawtooth wave (quieter)
                
                # Apply envelope to avoid clicks
                if len(t) > 2 * fade_samples:
                    envelope = np.ones_like(t)
                    envelope[:fade_samples] = fade_in
                    envelope[-fade_samples:] = fade_out
                    wave *= envelope
                
                row_audio += wave * (0.1 / len(frequencies))  # Scale by number of instruments
            
            audio[start_sample:end_sample] = row_audio
        
        return audio
    
    def create_baseline_sample(self, duration=5.0, output_file="baseline_sample.wav"):
        """Create a short baseline sample from normal data"""
        print("Creating baseline sample...")
        
        if self.is_anomaly is None:
            raise ValueError("Run detect_anomalies() first")
        
        # Get normal data (not anomalies)
        normal_data = self.data[~self.is_anomaly]
        
        # Sample for the specified duration
        rows_needed = int(duration / 0.1)  # 0.1 seconds per row
        if len(normal_data) > rows_needed:
            sample_data = normal_data.sample(n=rows_needed, random_state=42)
        else:
            sample_data = normal_data
        
        # Convert to frequencies and generate audio
        frequencies = self.data_to_frequencies(sample_data)
        audio = self.generate_audio(frequencies)
        
        # Normalize audio
        audio = audio / np.max(np.abs(audio))
        
        # Save as WAV file
        sf.write(output_file, audio, self.sample_rate)
        print(f"Baseline sample saved: {output_file} ({len(audio)/self.sample_rate:.1f}s)")
        
        return audio
    
    def create_anomaly_audio(self, output_file="anomalies.wav"):
        """Create full-length audio containing only anomalies"""
        print("Creating anomaly audio...")
        
        if self.is_anomaly is None:
            raise ValueError("Run detect_anomalies() first")
        
        # Get anomaly data
        anomaly_data = self.data[self.is_anomaly]
        
        if len(anomaly_data) == 0:
            print("No anomalies found!")
            return None
        
        print(f"Generating audio for {len(anomaly_data)} anomalies...")
        
        # Convert to frequencies and generate audio
        frequencies = self.data_to_frequencies(anomaly_data)
        audio = self.generate_audio(frequencies)
        
        # Normalize audio
        if len(audio) > 0:
            audio = audio / np.max(np.abs(audio))
        
        # Save as WAV file
        sf.write(output_file, audio, self.sample_rate)
        duration = len(audio) / self.sample_rate
        print(f"Anomaly audio saved: {output_file} ({duration:.1f}s, {len(anomaly_data)} anomalies)")
        
        return audio
    
    def process_folder(self, baseline_duration=5.0):
        """Complete pipeline: load data, detect anomalies, create audio files"""
        try:
            # Load data
            self.load_csv_files()
            
            # Detect anomalies
            self.detect_anomalies()
            
            # Create audio files
            baseline_audio = self.create_baseline_sample(duration=baseline_duration)
            anomaly_audio = self.create_anomaly_audio()
            
            # Print summary
            print(f"\nSummary:")
            print(f"Total rows processed: {len(self.data):,}")
            print(f"Total columns (instruments): {len(self.data.columns)}")
            print(f"Anomalies found: {np.sum(self.is_anomaly):,} ({np.sum(self.is_anomaly)/len(self.data)*100:.2f}%)")
            print(f"Files created: baseline_sample.wav, anomalies.wav")
            
            return {
                'total_rows': len(self.data),
                'total_columns': len(self.data.columns),
                'anomaly_count': np.sum(self.is_anomaly),
                'anomaly_percentage': np.sum(self.is_anomaly)/len(self.data)*100,
                'baseline_audio': baseline_audio,
                'anomaly_audio': anomaly_audio
            }
            
        except Exception as e:
            print(f"Error in processing: {e}")
            raise

# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = CSVMusicAnomalyDetector(
        folder_path="output",
        contamination=0.05  # Expect 5% anomalies
    )
    
    # Process all CSV files and create audio
    results = detector.process_folder(baseline_duration=5.0)