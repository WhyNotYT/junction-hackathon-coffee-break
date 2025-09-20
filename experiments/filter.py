import pandas as pd
import numpy as np
import wave
import struct
import os
import json
from scipy import signal
from scipy.fft import fft, ifft, fftfreq
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings

warnings.filterwarnings("ignore")

class AdaptiveAudioAnomalyFilter:
    """Enhanced audio-based anomaly detection with adaptive thresholding"""
    
    def __init__(self, output_dir="audio_analysis"):
        self.output_dir = output_dir
        self.sample_rate = 44100
        
    def read_wav_file(self, filename):
        """Read WAV file and return audio data as numpy array"""
        try:
            with wave.open(filename, 'r') as wav_file:
                frames = wav_file.readframes(-1)
                audio_data = struct.unpack('<' + ('h' * wav_file.getnframes()), frames)
                # Convert to float and normalize
                audio_data = np.array(audio_data, dtype=np.float32) / 32767.0
                return audio_data
        except Exception as e:
            print(f"❌ Error reading {filename}: {e}")
            return np.array([])
    
    def write_wav_file(self, data, filename):
        """Write numpy array as WAV file"""
        try:
            # Normalize and convert to 16-bit integers
            data_normalized = np.clip(data, -1.0, 1.0)
            audio_data_int = (data_normalized * 32767).astype(np.int16)
            
            with wave.open(filename, 'w') as wav_file:
                wav_file.setparams((1, 2, self.sample_rate, len(audio_data_int), 'NONE', 'not compressed'))
                for sample in audio_data_int:
                    wav_file.writeframes(struct.pack('<h', sample))
            
            print(f"✅ Saved: {filename}")
            return True
        except Exception as e:
            print(f"❌ Error writing {filename}: {e}")
            return False
    
    def adaptive_noise_reduction(self, signal_audio, noise_audio):
        """Enhanced adaptive noise reduction"""
        if len(signal_audio) == 0:
            return signal_audio
        
        # Method 1: Spectral subtraction with adaptive parameters
        cleaned_1 = self.spectral_subtraction_adaptive(signal_audio, noise_audio)
        
        # Method 2: Wiener filtering
        cleaned_2 = self.wiener_filter(signal_audio, noise_audio, noise_factor=0.15)
        
        # Method 3: Median filtering for impulse noise
        cleaned_3 = self.median_filter_denoising(signal_audio, noise_audio)
        
        # Combine methods with optimized weights
        cleaned_combined = (0.5 * cleaned_1 + 0.3 * cleaned_2 + 0.2 * cleaned_3)
        
        return cleaned_combined
    
    def spectral_subtraction_adaptive(self, signal_audio, noise_audio, alpha=1.8, beta=0.05):
        """Enhanced spectral subtraction with adaptive parameters"""
        if len(signal_audio) == 0 or len(noise_audio) == 0:
            return signal_audio
        
        # Ensure both signals have the same length
        min_len = min(len(signal_audio), len(noise_audio))
        signal_audio = signal_audio[:min_len]
        noise_audio = noise_audio[:min_len]
        
        # Apply windowing
        window = np.hanning(len(signal_audio))
        signal_windowed = signal_audio * window
        noise_windowed = noise_audio * window
        
        # Compute FFT
        signal_fft = fft(signal_windowed)
        noise_fft = fft(noise_windowed)
        
        # Compute magnitude spectra
        signal_magnitude = np.abs(signal_fft)
        noise_magnitude = np.abs(noise_fft)
        signal_phase = np.angle(signal_fft)
        
        # Adaptive noise estimation
        noise_spectrum = np.maximum(noise_magnitude, 0.1 * signal_magnitude)
        
        # Adaptive spectral subtraction
        cleaned_magnitude = signal_magnitude - alpha * noise_spectrum
        
        # Dynamic spectral floor
        spectral_floor = beta * signal_magnitude
        cleaned_magnitude = np.maximum(cleaned_magnitude, spectral_floor)
        
        # Reconstruct complex spectrum
        cleaned_fft = cleaned_magnitude * np.exp(1j * signal_phase)
        
        # Inverse FFT
        cleaned_audio = np.real(ifft(cleaned_fft))
        
        return cleaned_audio
    
    def wiener_filter(self, signal_audio, noise_audio, noise_factor=0.15):
        """Enhanced Wiener filtering"""
        if len(signal_audio) == 0 or len(noise_audio) == 0:
            return signal_audio
        
        min_len = min(len(signal_audio), len(noise_audio))
        signal_audio = signal_audio[:min_len]
        noise_audio = noise_audio[:min_len]
        
        # Compute power spectral densities
        signal_fft = fft(signal_audio)
        noise_fft = fft(noise_audio)
        
        signal_psd = np.abs(signal_fft) ** 2
        noise_psd = np.abs(noise_fft) ** 2
        
        # Better noise PSD estimation
        noise_psd_est = np.maximum(np.mean(noise_psd) * noise_factor, 0.01 * signal_psd)
        
        # Wiener filter
        wiener_filter_coeff = signal_psd / (signal_psd + noise_psd_est)
        
        # Apply filter
        filtered_fft = signal_fft * wiener_filter_coeff
        filtered_audio = np.real(ifft(filtered_fft))
        
        return filtered_audio
    
    def median_filter_denoising(self, signal_audio, noise_audio, kernel_size=5):
        """Median filtering for impulse noise removal"""
        if len(signal_audio) == 0:
            return signal_audio
        
        # Apply median filter to remove impulse-like anomalies
        from scipy.ndimage import median_filter
        filtered_audio = median_filter(signal_audio, size=kernel_size, mode='reflect')
        
        return filtered_audio
    
    def calculate_adaptive_threshold(self, indices, shift_scores, method='spline', window_size=5000, 
                                   threshold_factor=2.5, min_threshold=0.01):
        """
        Calculate adaptive threshold that follows the data pattern
        
        Args:
            indices: array of row indices
            shift_scores: array of shift scores
            method: 'spline', 'rolling', 'gaussian', or 'polynomial'
            window_size: size of rolling window for smoothing
            threshold_factor: multiplier for local standard deviation
            min_threshold: minimum threshold value
        
        Returns:
            array of adaptive thresholds for each point
        """
        if len(indices) == 0 or len(shift_scores) == 0:
            return np.array([])
        
        print(f"🎯 Calculating adaptive threshold using method: {method}")
        
        # Sort by indices for proper ordering
        sorted_idx = np.argsort(indices)
        sorted_indices = np.array(indices)[sorted_idx]
        sorted_scores = np.array(shift_scores)[sorted_idx]
        
        if method == 'spline':
            # Use spline interpolation to create smooth baseline
            try:
                # Use quantile-based baseline to avoid anomaly influence
                baseline_scores = np.percentile(sorted_scores, 25)  # 25th percentile as baseline
                
                # Create bins for local statistics
                n_bins = max(10, len(sorted_scores) // 1000)
                bins = np.linspace(0, len(sorted_scores)-1, n_bins)
                bin_indices = np.digitize(range(len(sorted_scores)), bins) - 1
                
                baseline_values = []
                local_stds = []
                bin_centers = []
                
                for i in range(n_bins-1):
                    mask = bin_indices == i
                    if np.sum(mask) > 0:
                        local_scores = sorted_scores[mask]
                        # Use robust statistics
                        local_baseline = np.percentile(local_scores, 30)
                        local_std = np.std(local_scores)
                        baseline_values.append(local_baseline)
                        local_stds.append(local_std)
                        bin_centers.append(np.mean(sorted_indices[mask]))
                
                if len(baseline_values) > 3:
                    # Interpolate baseline and std
                    baseline_spline = UnivariateSpline(bin_centers, baseline_values, s=len(baseline_values)*0.1)
                    std_spline = UnivariateSpline(bin_centers, local_stds, s=len(local_stds)*0.1)
                    
                    adaptive_baseline = baseline_spline(sorted_indices)
                    adaptive_std = std_spline(sorted_indices)
                    
                    # Calculate adaptive threshold
                    adaptive_thresholds = adaptive_baseline + threshold_factor * adaptive_std
                else:
                    # Fallback to global statistics
                    adaptive_thresholds = np.full_like(sorted_scores, 
                                                     np.mean(sorted_scores) + threshold_factor * np.std(sorted_scores))
                
            except Exception as e:
                print(f"⚠️ Spline method failed: {e}, using fallback")
                adaptive_thresholds = np.full_like(sorted_scores, 
                                                 np.mean(sorted_scores) + threshold_factor * np.std(sorted_scores))
        
        elif method == 'rolling':
            # Rolling window approach
            from pandas import Series
            score_series = Series(sorted_scores)
            
            # Calculate rolling statistics
            rolling_mean = score_series.rolling(window=min(window_size, len(sorted_scores)//2), 
                                              center=True, min_periods=10).mean()
            rolling_std = score_series.rolling(window=min(window_size, len(sorted_scores)//2), 
                                             center=True, min_periods=10).std()
            
            # Fill NaN values at edges
            rolling_mean = rolling_mean.fillna(method='bfill').fillna(method='ffill')
            rolling_std = rolling_std.fillna(method='bfill').fillna(method='ffill')
            
            adaptive_thresholds = rolling_mean + threshold_factor * rolling_std
        
        elif method == 'gaussian':
            # Gaussian smoothing approach
            sigma = len(sorted_scores) / 20  # Adaptive sigma
            smoothed_scores = gaussian_filter1d(sorted_scores, sigma=sigma)
            
            # Calculate local deviations
            deviations = np.abs(sorted_scores - smoothed_scores)
            smoothed_deviations = gaussian_filter1d(deviations, sigma=sigma)
            
            adaptive_thresholds = smoothed_scores + threshold_factor * smoothed_deviations
        
        elif method == 'polynomial':
            # Polynomial fit approach
            try:
                degree = min(6, len(sorted_scores) // 1000)  # Adaptive degree
                poly_coeffs = np.polyfit(range(len(sorted_scores)), sorted_scores, degree)
                baseline_fit = np.polyval(poly_coeffs, range(len(sorted_scores)))
                
                residuals = sorted_scores - baseline_fit
                local_std = np.std(residuals)
                
                adaptive_thresholds = baseline_fit + threshold_factor * local_std
            except Exception as e:
                print(f"⚠️ Polynomial method failed: {e}, using fallback")
                adaptive_thresholds = np.full_like(sorted_scores, 
                                                 np.mean(sorted_scores) + threshold_factor * np.std(sorted_scores))
        
        # Ensure minimum threshold
        adaptive_thresholds = np.maximum(adaptive_thresholds, min_threshold)
        
        # Restore original order
        original_order_thresholds = np.zeros_like(adaptive_thresholds)
        original_order_thresholds[sorted_idx] = adaptive_thresholds
        
        return original_order_thresholds
    
    def identify_anomalous_samples_adaptive(self, csv_file, flagged_file, method='spline', 
                                          threshold_factor=2.5, visualization=True):
        """
        Enhanced anomaly identification with adaptive thresholding
        """
        print(f"🔍 Identifying anomalous samples with adaptive thresholding...")
        
        # Load original data
        try:
            df = pd.read_csv(csv_file)
            print(f"📊 Loaded original data: {len(df)} rows")
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            return set()
        
        # Get numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        print(f"🔢 Processing {len(numeric_columns)} numeric columns")
        
        # Collect shift data for all columns
        all_sample_shifts = defaultdict(list)
        
        # Process each column
        for column in numeric_columns:
            column_result = self.process_column_audio(column)
            if column_result is None:
                continue
            
            # Map audio shifts to data sample indices
            data_shifts = self.map_audio_shifts_to_data_indices(
                column_result['shifts'], len(df)
            )
            
            # Store shifts for each row
            for i, shift in enumerate(data_shifts):
                all_sample_shifts[i].append(shift)
        
        # Calculate combined shift score for each row
        shift_scores = {}
        indices = []
        
        for row_idx, shifts_list in all_sample_shifts.items():
            if len(shifts_list) == 0:
                continue
            
            # Combine shifts from multiple columns
            combined_shift = np.mean(shifts_list)
            shift_scores[row_idx] = combined_shift
            indices.append(row_idx)
        
        if len(shift_scores) == 0:
            print("⚠️ No shift data available")
            return set()
        
        # Calculate adaptive thresholds
        scores_array = [shift_scores[idx] for idx in indices]
        adaptive_thresholds = self.calculate_adaptive_threshold(
            indices, scores_array, method=method, threshold_factor=threshold_factor
        )
        
        # Identify anomalous samples
        anomalous_indices = set()
        threshold_violations = []
        
        for i, (row_idx, threshold) in enumerate(zip(indices, adaptive_thresholds)):
            score = shift_scores[row_idx]
            if score > threshold:
                anomalous_indices.add(row_idx)
                threshold_violations.append((row_idx, score, threshold, score - threshold))
        
        print(f"🚨 Identified {len(anomalous_indices)} anomalous samples")
        
        # Save enhanced analysis results
        analysis_data = {
            'method': method,
            'threshold_factor': threshold_factor,
            'adaptive_thresholds': {str(idx): float(thresh) for idx, thresh in zip(indices, adaptive_thresholds)},
            'anomalous_indices': list(anomalous_indices),
            'shift_scores': {str(k): float(v) for k, v in shift_scores.items()},
            'threshold_violations': [(int(idx), float(score), float(thresh), float(violation)) 
                                   for idx, score, thresh, violation in threshold_violations],
            'total_samples': len(df),
            'total_anomalies': len(anomalous_indices),
            'stats': {
                'mean_shift': float(np.mean(scores_array)),
                'std_shift': float(np.std(scores_array)),
                'min_shift': float(np.min(scores_array)),
                'max_shift': float(np.max(scores_array))
            }
        }
        
        analysis_file = os.path.join(self.output_dir, "adaptive_shift_analysis.json")
        try:
            with open(analysis_file, 'w') as f:
                json.dump(analysis_data, f, indent=2)
            print(f"📝 Analysis saved to: {analysis_file}")
        except Exception as e:
            print(f"⚠️ Could not save analysis: {e}")
        
        # Create enhanced visualization
        if visualization:
            self.create_adaptive_threshold_visualization(indices, scores_array, adaptive_thresholds, 
                                                       anomalous_indices, method)
        
        return anomalous_indices
    
    def create_adaptive_threshold_visualization(self, indices, scores, thresholds, anomalous_indices, method):
        """Create enhanced visualization with adaptive threshold"""
        try:
            plt.figure(figsize=(15, 8))
            
            # Sort for plotting
            sorted_idx = np.argsort(indices)
            sorted_indices = np.array(indices)[sorted_idx]
            sorted_scores = np.array(scores)[sorted_idx]
            sorted_thresholds = np.array(thresholds)[sorted_idx]
            
            # Plot all points
            plt.scatter(sorted_indices, sorted_scores, alpha=0.4, s=15, color='blue', label='Normal samples')
            
            # Highlight anomalous points
            anomaly_mask = np.isin(sorted_indices, list(anomalous_indices))
            if np.any(anomaly_mask):
                plt.scatter(sorted_indices[anomaly_mask], sorted_scores[anomaly_mask], 
                           color='red', s=25, alpha=0.8, label=f'Anomalies ({len(anomalous_indices)})')
            
            # Plot adaptive threshold
            plt.plot(sorted_indices, sorted_thresholds, 'r--', linewidth=2, 
                    label=f'Adaptive Threshold ({method})')
            
            # Fill area above threshold
            plt.fill_between(sorted_indices, sorted_thresholds, np.max(sorted_scores), 
                           alpha=0.1, color='red', label='Anomaly Region')
            
            plt.xlabel('Row Index')
            plt.ylabel('Shift Score')
            plt.title(f'Adaptive Audio-Based Anomaly Detection\nMethod: {method}, Anomalies: {len(anomalous_indices)}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save plot
            viz_file = os.path.join(self.output_dir, f"adaptive_threshold_analysis_{method}.png")
            plt.savefig(viz_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"📈 Enhanced visualization saved to: {viz_file}")
            
            # Create comparison of different methods
            self.create_method_comparison_plot(indices, scores)
            
        except Exception as e:
            print(f"⚠️ Could not create visualization: {e}")
    
    def create_method_comparison_plot(self, indices, scores):
        """Compare different adaptive threshold methods"""
        try:
            methods = ['spline', 'rolling', 'gaussian', 'polynomial']
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()
            
            sorted_idx = np.argsort(indices)
            sorted_indices = np.array(indices)[sorted_idx]
            sorted_scores = np.array(scores)[sorted_idx]
            
            for i, method in enumerate(methods):
                ax = axes[i]
                
                try:
                    # Calculate threshold for this method
                    thresholds = self.calculate_adaptive_threshold(indices, scores, method=method)
                    sorted_thresholds = np.array(thresholds)[sorted_idx]
                    
                    # Plot
                    ax.scatter(sorted_indices, sorted_scores, alpha=0.3, s=10, color='blue')
                    ax.plot(sorted_indices, sorted_thresholds, 'r-', linewidth=2, label=f'{method} threshold')
                    ax.fill_between(sorted_indices, sorted_thresholds, np.max(sorted_scores), 
                                  alpha=0.1, color='red')
                    
                    # Count anomalies for this method
                    anomaly_count = np.sum(sorted_scores > sorted_thresholds)
                    ax.set_title(f'{method.title()} Method\nAnomalies: {anomaly_count}')
                    ax.set_xlabel('Row Index')
                    ax.set_ylabel('Shift Score')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                except Exception as e:
                    ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{method.title()} Method - Failed')
            
            plt.tight_layout()
            comparison_file = os.path.join(self.output_dir, "threshold_methods_comparison.png")
            plt.savefig(comparison_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Method comparison saved to: {comparison_file}")
            
        except Exception as e:
            print(f"⚠️ Could not create method comparison: {e}")
    
    def process_column_audio(self, column_name):
        """Process audio files for a specific column and identify anomalies"""
        print(f"\n🎵 Processing audio for column: {column_name}")
        
        # File paths
        standard_file = os.path.join(self.output_dir, f"{column_name}_standard.wav")
        anomalies_file = os.path.join(self.output_dir, f"{column_name}_anomalies_only.wav")
        
        # Check if files exist
        if not os.path.exists(standard_file) or not os.path.exists(anomalies_file):
            print(f"⚠️ Audio files not found for {column_name}")
            return None
        
        # Read audio files
        standard_audio = self.read_wav_file(standard_file)
        anomalies_audio = self.read_wav_file(anomalies_file)
        
        if len(standard_audio) == 0:
            return None
        
        # Apply enhanced noise reduction
        cleaned_audio = self.adaptive_noise_reduction(standard_audio, anomalies_audio)
        
        # Save cleaned audio
        cleaned_filename = os.path.join(self.output_dir, f"{column_name}_adaptive_cleaned.wav")
        self.write_wav_file(cleaned_audio, cleaned_filename)
        
        # Calculate shifts
        shifts = self.calculate_sample_shifts(standard_audio, cleaned_audio)
        
        return {
            'column': column_name,
            'shifts': shifts,
            'standard_audio': standard_audio,
            'cleaned_audio': cleaned_audio
        }
    
    def calculate_sample_shifts(self, original_audio, cleaned_audio):
        """Calculate how much each sample shifted after noise reduction"""
        if len(original_audio) != len(cleaned_audio):
            min_len = min(len(original_audio), len(cleaned_audio))
            original_audio = original_audio[:min_len]
            cleaned_audio = cleaned_audio[:min_len]
        
        shifts = np.abs(original_audio - cleaned_audio)
        return shifts
    
    def map_audio_shifts_to_data_indices(self, shifts, original_data_length):
        """Map audio sample shifts back to original data indices"""
        if len(shifts) == 0:
            return np.array([])
        
        audio_length = len(shifts)
        data_shifts = np.zeros(original_data_length)
        
        for i in range(original_data_length):
            # Map data index to audio index
            audio_idx = int((i / original_data_length) * audio_length)
            audio_idx = min(audio_idx, audio_length - 1)
            
            # Average shifts in a window
            window_size = max(1, audio_length // original_data_length)
            start_idx = max(0, audio_idx - window_size // 2)
            end_idx = min(audio_length, audio_idx + window_size // 2 + 1)
            
            data_shifts[i] = np.mean(shifts[start_idx:end_idx])
        
        return data_shifts
    
    def create_clean_csv(self, csv_file, anomalous_indices, output_file="cleaned_data_adaptive.csv"):
        """Create a clean CSV by removing anomalous samples"""
        print(f"🧹 Creating clean CSV with adaptive method...")
        
        try:
            # Load original data
            df = pd.read_csv(csv_file)
            original_count = len(df)
            
            # Remove anomalous rows
            df_clean = df.drop(index=list(anomalous_indices)).reset_index(drop=True)
            clean_count = len(df_clean)
            removed_count = original_count - clean_count
            
            # Save cleaned data
            df_clean.to_csv(output_file, index=False)
            
            print(f"✅ Adaptive clean CSV created: {output_file}")
            print(f"📊 Original rows: {original_count}")
            print(f"📊 Removed rows: {removed_count}")
            print(f"📊 Clean rows: {clean_count}")
            print(f"📊 Removal rate: {(removed_count/original_count)*100:.2f}%")
            
            # Create enhanced summary
            summary_file = output_file.replace('.csv', '_summary.txt')
            with open(summary_file, 'w') as f:
                f.write("Adaptive Audio-Based Data Cleaning Summary\n")
                f.write("=========================================\n\n")
                f.write(f"Original file: {csv_file}\n")
                f.write(f"Cleaned file: {output_file}\n")
                f.write(f"Method: Adaptive threshold with audio noise reduction\n")
                f.write(f"Original rows: {original_count}\n")
                f.write(f"Removed rows: {removed_count}\n")
                f.write(f"Clean rows: {clean_count}\n")
                f.write(f"Removal rate: {(removed_count/original_count)*100:.2f}%\n\n")
                f.write("Removed row indices (first 100):\n")
                for idx in sorted(list(anomalous_indices)[:100]):
                    f.write(f"{idx}\n")
                if len(anomalous_indices) > 100:
                    f.write(f"... and {len(anomalous_indices) - 100} more\n")
            
            return df_clean
            
        except Exception as e:
            print(f"❌ Error creating clean CSV: {e}")
            return None

def main():
    """Main function with enhanced adaptive processing"""
    
    # Configuration
    CSV_FILE = "Datasets/GiveMeSomeCredit/cs-training.csv"
    FLAGGED_FILE = "flagged.txt"
    AUDIO_DIR = "audio_analysis"
    OUTPUT_CSV = "cleaned_data_adaptive_audio.csv"
    
    # Adaptive threshold parameters
    THRESHOLD_METHOD = 'spline'  # 'spline', 'rolling', 'gaussian', 'polynomial'
    THRESHOLD_FACTOR = 2.5  # Adjust sensitivity
    
    print("🎵 Adaptive Audio-Based Anomaly Removal System")
    print("=" * 50)
    print(f"🎯 Using adaptive threshold method: {THRESHOLD_METHOD}")
    print(f"⚙️ Threshold factor: {THRESHOLD_FACTOR}")
    
    # Initialize the enhanced filter
    audio_filter = AdaptiveAudioAnomalyFilter(output_dir=AUDIO_DIR)
    
    # Check if audio files exist
    if not os.path.exists(AUDIO_DIR):
        print(f"❌ Audio directory not found: {AUDIO_DIR}")
        print("Please run the audio converter script first!")
        return
    
    # Identify anomalous samples using adaptive method
    anomalous_indices = audio_filter.identify_anomalous_samples_adaptive(
        csv_file=CSV_FILE,
        flagged_file=FLAGGED_FILE,
        method=THRESHOLD_METHOD,
        threshold_factor=THRESHOLD_FACTOR,
        visualization=True
    )
    
    if len(anomalous_indices) == 0:
        print("⚠️ No anomalies detected through adaptive audio analysis")
        return
    
    # Create clean CSV
    cleaned_df = audio_filter.create_clean_csv(
        csv_file=CSV_FILE,
        anomalous_indices=anomalous_indices,
        output_file=OUTPUT_CSV
    )
    
    if cleaned_df is not None:
        print(f"\n🎉 Adaptive process completed successfully!")
        print(f"📁 Clean data saved to: {OUTPUT_CSV}")
        print(f"🔊 Enhanced analysis files in: {AUDIO_DIR}")
        print(f"📈 Visualizations show adaptive threshold following data curve")

if __name__ == "__main__":
    main()