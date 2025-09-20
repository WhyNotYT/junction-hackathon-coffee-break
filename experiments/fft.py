import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class AnomalyFFTAnalyzer:
    def __init__(self, flagged_file, original_csv):
        """
        Initialize the FFT analyzer for anomaly detection
        
        Args:
            flagged_file: Path to the flagged.txt file with anomaly details
            original_csv: Path to the original CSV dataset
        """
        self.flagged_file = flagged_file
        self.original_csv = original_csv
        self.flagged_data = []
        self.original_df = None
        self.anomaly_indices = []
        
    def load_data(self):
        """Load both flagged data and original CSV"""
        print("📂 Loading flagged anomaly data...")
        
        # Load flagged data
        try:
            with open(self.flagged_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        record = json.loads(line.strip())
                        self.flagged_data.append(record)
                    except json.JSONDecodeError as e:
                        print(f"⚠️ Skipping invalid JSON on line {line_num}: {e}")
            
            print(f"✅ Loaded {len(self.flagged_data)} flagged records")
            
        except FileNotFoundError:
            print(f"❌ Flagged file {self.flagged_file} not found")
            return False
        
        # Load original CSV
        print("📂 Loading original dataset...")
        try:
            self.original_df = pd.read_csv(self.original_csv)
            print(f"✅ Loaded original dataset: {self.original_df.shape[0]} rows, {self.original_df.shape[1]} columns")
            
            # Extract anomaly indices
            self.anomaly_indices = [record['row_index'] for record in self.flagged_data]
            print(f"🎯 Found {len(self.anomaly_indices)} anomalous row indices")
            
            return True
            
        except FileNotFoundError:
            print(f"❌ Original CSV file {self.original_csv} not found")
            return False
    
    def prepare_data_for_fft(self, columns=None):
        """
        Prepare data for FFT analysis
        
        Args:
            columns: List of columns to analyze (None = all numeric columns)
        """
        # Select numeric columns
        numeric_df = self.original_df.select_dtypes(include=[np.number])
        
        if columns is None:
            # Use continuous columns (exclude binary)
            columns = [col for col in numeric_df.columns if numeric_df[col].nunique() > 2]
        
        print(f"🔢 Analyzing columns: {columns}")
        
        # Fill missing values with median
        analysis_df = numeric_df[columns].fillna(numeric_df[columns].median())
        
        # Separate anomalous and normal data
        anomaly_mask = self.original_df.index.isin(self.anomaly_indices)
        
        self.anomalous_data = analysis_df[anomaly_mask]
        self.normal_data = analysis_df[~anomaly_mask]
        
        print(f"📊 Anomalous samples: {len(self.anomalous_data)}")
        print(f"📊 Normal samples: {len(self.normal_data)}")
        
        return columns
    
    def compute_fft_spectrum(self, data, sampling_rate=1.0):
        """
        Compute FFT spectrum for given data
        
        Args:
            data: DataFrame with numeric data
            sampling_rate: Sampling rate for FFT
        
        Returns:
            freqs: Frequency array
            magnitude_spectrum: Average magnitude spectrum across all columns
            phase_spectrum: Average phase spectrum across all columns
        """
        n_samples = len(data)
        if n_samples < 2:
            return None, None, None
        
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        
        # Compute FFT for each column and average
        all_magnitudes = []
        all_phases = []
        
        for col_idx in range(scaled_data.shape[1]):
            column_data = scaled_data[:, col_idx]
            
            # Compute FFT
            fft_values = fft(column_data)
            
            # Compute magnitude and phase
            magnitude = np.abs(fft_values)
            phase = np.angle(fft_values)
            
            all_magnitudes.append(magnitude)
            all_phases.append(phase)
        
        # Average across all columns
        avg_magnitude = np.mean(all_magnitudes, axis=0)
        avg_phase = np.mean(all_phases, axis=0)
        
        # Generate frequency array
        freqs = fftfreq(n_samples, d=1/sampling_rate)
        
        return freqs, avg_magnitude, avg_phase
    
    def plot_fft_comparison(self, columns=None, sampling_rate=1.0):
        """
        Plot FFT comparison between anomalous and normal data
        
        Args:
            columns: Columns to analyze
            sampling_rate: Sampling rate for FFT
        """
        columns = self.prepare_data_for_fft(columns)
        
        # Compute FFT for both datasets
        print("🔄 Computing FFT for anomalous data...")
        freqs_anom, mag_anom, phase_anom = self.compute_fft_spectrum(self.anomalous_data, sampling_rate)
        
        print("🔄 Computing FFT for normal data...")
        freqs_norm, mag_norm, phase_norm = self.compute_fft_spectrum(self.normal_data, sampling_rate)
        
        if freqs_anom is None or freqs_norm is None:
            print("❌ Insufficient data for FFT analysis")
            return
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('FFT Analysis: Anomalous vs Normal Data', fontsize=16, fontweight='bold')
        
        # 1. Magnitude Spectrum Comparison
        ax1 = axes[0, 0]
        
        # Only plot positive frequencies for clarity
        positive_freq_mask_anom = freqs_anom >= 0
        positive_freq_mask_norm = freqs_norm >= 0
        
        ax1.semilogy(freqs_anom[positive_freq_mask_anom], mag_anom[positive_freq_mask_anom], 
                    'r-', alpha=0.7, linewidth=2, label=f'Anomalous (n={len(self.anomalous_data)})')
        ax1.semilogy(freqs_norm[positive_freq_mask_norm], mag_norm[positive_freq_mask_norm], 
                    'b-', alpha=0.7, linewidth=2, label=f'Normal (n={len(self.normal_data)})')
        
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Magnitude (Log Scale)')
        ax1.set_title('Magnitude Spectrum Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Phase Spectrum Comparison
        ax2 = axes[0, 1]
        ax2.plot(freqs_anom[positive_freq_mask_anom], phase_anom[positive_freq_mask_anom], 
                'r-', alpha=0.7, linewidth=2, label='Anomalous')
        ax2.plot(freqs_norm[positive_freq_mask_norm], phase_norm[positive_freq_mask_norm], 
                'b-', alpha=0.7, linewidth=2, label='Normal')
        
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Phase (Radians)')
        ax2.set_title('Phase Spectrum Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Magnitude Difference
        ax3 = axes[1, 0]
        
        # Interpolate to same frequency grid for comparison
        min_len = min(len(freqs_anom), len(freqs_norm))
        freq_common = freqs_anom[:min_len]
        mag_diff = mag_anom[:min_len] - mag_norm[:min_len]
        
        positive_mask = freq_common >= 0
        ax3.plot(freq_common[positive_mask], mag_diff[positive_mask], 
                'g-', linewidth=2, label='Anomalous - Normal')
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        ax3.set_xlabel('Frequency (Hz)')
        ax3.set_ylabel('Magnitude Difference')
        ax3.set_title('Spectral Difference (Anomalous - Normal)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Power Spectral Density
        ax4 = axes[1, 1]
        
        # Compute power spectral density (magnitude squared)
        psd_anom = mag_anom ** 2
        psd_norm = mag_norm ** 2
        
        ax4.semilogy(freqs_anom[positive_freq_mask_anom], psd_anom[positive_freq_mask_anom], 
                    'r-', alpha=0.7, linewidth=2, label='Anomalous')
        ax4.semilogy(freqs_norm[positive_freq_mask_norm], psd_norm[positive_freq_mask_norm], 
                    'b-', alpha=0.7, linewidth=2, label='Normal')
        
        ax4.set_xlabel('Frequency (Hz)')
        ax4.set_ylabel('Power Spectral Density (Log Scale)')
        ax4.set_title('Power Spectral Density Comparison')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_column_specific_fft(self, column_name, sampling_rate=1.0):
        """
        Plot FFT analysis for a specific column
        
        Args:
            column_name: Name of the column to analyze
            sampling_rate: Sampling rate for FFT
        """
        if column_name not in self.original_df.columns:
            print(f"❌ Column '{column_name}' not found in dataset")
            return
        
        print(f"🔍 Analyzing FFT for column: {column_name}")
        
        # Prepare data for single column
        column_data = self.original_df[column_name].fillna(self.original_df[column_name].median())
        
        anomaly_mask = self.original_df.index.isin(self.anomaly_indices)
        anom_data = column_data[anomaly_mask].values
        norm_data = column_data[~anomaly_mask].values
        
        # Standardize
        scaler = StandardScaler()
        anom_data_scaled = scaler.fit_transform(anom_data.reshape(-1, 1)).flatten()
        norm_data_scaled = scaler.fit_transform(norm_data.reshape(-1, 1)).flatten()
        
        # Compute FFT
        fft_anom = fft(anom_data_scaled)
        fft_norm = fft(norm_data_scaled)
        
        freqs_anom = fftfreq(len(anom_data_scaled), d=1/sampling_rate)
        freqs_norm = fftfreq(len(norm_data_scaled), d=1/sampling_rate)
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'FFT Analysis for Column: {column_name}', fontsize=14, fontweight='bold')
        
        # Magnitude spectrum
        positive_mask_anom = freqs_anom >= 0
        positive_mask_norm = freqs_norm >= 0
        
        ax1.semilogy(freqs_anom[positive_mask_anom], np.abs(fft_anom[positive_mask_anom]), 
                    'r-', alpha=0.7, linewidth=2, label=f'Anomalous (n={len(anom_data)})')
        ax1.semilogy(freqs_norm[positive_mask_norm], np.abs(fft_norm[positive_mask_norm]), 
                    'b-', alpha=0.7, linewidth=2, label=f'Normal (n={len(norm_data)})')
        
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Magnitude (Log Scale)')
        ax1.set_title('Magnitude Spectrum')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Phase spectrum
        ax2.plot(freqs_anom[positive_mask_anom], np.angle(fft_anom[positive_mask_anom]), 
                'r-', alpha=0.7, linewidth=2, label='Anomalous')
        ax2.plot(freqs_norm[positive_mask_norm], np.angle(fft_norm[positive_mask_norm]), 
                'b-', alpha=0.7, linewidth=2, label='Normal')
        
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Phase (Radians)')
        ax2.set_title('Phase Spectrum')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def analyze_frequency_domain_differences(self, top_n=5):
        """
        Analyze which frequency components differ most between anomalous and normal data
        
        Args:
            top_n: Number of top frequency differences to report
        """
        columns = self.prepare_data_for_fft()
        
        # Compute FFT for both datasets
        freqs_anom, mag_anom, phase_anom = self.compute_fft_spectrum(self.anomalous_data)
        freqs_norm, mag_norm, phase_norm = self.compute_fft_spectrum(self.normal_data)
        
        if freqs_anom is None or freqs_norm is None:
            print("❌ Insufficient data for frequency analysis")
            return
        
        # Calculate magnitude differences
        min_len = min(len(mag_anom), len(mag_norm))
        mag_diff = np.abs(mag_anom[:min_len] - mag_norm[:min_len])
        freqs_common = freqs_anom[:min_len]
        
        # Find top frequency differences (positive frequencies only)
        positive_mask = freqs_common >= 0
        positive_freqs = freqs_common[positive_mask]
        positive_diffs = mag_diff[positive_mask]
        
        # Get top differences
        top_indices = np.argsort(positive_diffs)[-top_n:][::-1]
        
        print(f"\n🔍 Top {top_n} Frequency Domain Differences:")
        print("=" * 50)
        
        for i, idx in enumerate(top_indices, 1):
            freq = positive_freqs[idx]
            diff = positive_diffs[idx]
            anom_mag = mag_anom[positive_mask][idx] if idx < len(mag_anom[positive_mask]) else 0
            norm_mag = mag_norm[positive_mask][idx] if idx < len(mag_norm[positive_mask]) else 0
            
            print(f"{i}. Frequency: {freq:.4f} Hz")
            print(f"   Magnitude Difference: {diff:.4f}")
            print(f"   Anomalous Magnitude: {anom_mag:.4f}")
            print(f"   Normal Magnitude: {norm_mag:.4f}")
            print(f"   Ratio (Anom/Normal): {anom_mag/norm_mag if norm_mag != 0 else 'inf':.2f}")
            print()

def main():
    """Main function to run the FFT analysis"""
    
    # Configuration
    FLAGGED_FILE = "flagged.txt"  # Path to your flagged.txt file
    ORIGINAL_CSV = "Datasets/GiveMeSomeCredit/cs-training.csv"  # Path to your original CSV
    SAMPLING_RATE = 1.0  # Adjust based on your data characteristics
    
    print("🚀 Starting FFT Anomaly Analysis")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = AnomalyFFTAnalyzer(FLAGGED_FILE, ORIGINAL_CSV)
    
    # Load data
    if not analyzer.load_data():
        return
    
    # Perform comprehensive FFT analysis
    print("\n📊 Creating FFT comparison plots...")
    analyzer.plot_fft_comparison(sampling_rate=SAMPLING_RATE)
    
    # Analyze frequency domain differences
    print("\n🔍 Analyzing frequency domain differences...")
    analyzer.analyze_frequency_domain_differences(top_n=5)
    
    # Example: Analyze specific columns that were frequently flagged
    print("\n📈 Analyzing most frequently flagged columns...")
    
    # Count flagged columns
    from collections import Counter
    flagged_columns = []
    for record in analyzer.flagged_data:
        flagged_columns.extend(record.get('flagged_columns', []))
    
    col_counts = Counter(flagged_columns)
    most_flagged = col_counts.most_common(3)
    
    print(f"Most frequently flagged columns: {[col for col, count in most_flagged]}")
    
    # Plot FFT for most flagged columns
    for col, count in most_flagged:
        if col in analyzer.original_df.columns:
            print(f"\n🔍 FFT Analysis for '{col}' (flagged {count} times):")
            analyzer.plot_column_specific_fft(col, sampling_rate=SAMPLING_RATE)
    
    print("\n✅ FFT Analysis Complete!")

if __name__ == "__main__":
    main()