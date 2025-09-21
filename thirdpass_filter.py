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
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

warnings.filterwarnings("ignore")

class AudioAnomalyProcessor:
    """Enhanced audio-based anomaly processor using flagged references"""
    
    def __init__(self, output_dir="audio_analysis"):
        self.output_dir = output_dir
        self.sample_rate = 44100
        
    def load_flagged_indices(self, flagged_file):
        """Load flagged indices from file"""
        flagged_indices = set()
        try:
            with open(flagged_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and line.isdigit():
                        flagged_indices.add(int(line))
            print(f"📋 Loaded {len(flagged_indices)} flagged indices")
        except Exception as e:
            print(f"⚠️ Error loading flagged file: {e}")
        return flagged_indices
        
    def read_wav_file(self, filename):
        """Read WAV file and return audio data as numpy array"""
        try:
            with wave.open(filename, 'r') as wav_file:
                frames = wav_file.readframes(-1)
                audio_data = struct.unpack('<' + ('h' * wav_file.getnframes()), frames)
                audio_data = np.array(audio_data, dtype=np.float32) / 32767.0
                return audio_data
        except Exception as e:
            print(f"❌ Error reading {filename}: {e}")
            return np.array([])
    
    def write_wav_file(self, data, filename):
        """Write numpy array as WAV file"""
        try:
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
    
    def advanced_spectral_analysis(self, signal_audio, noise_audio):
        """Advanced spectral analysis to identify anomaly patterns"""
        if len(signal_audio) == 0 or len(noise_audio) == 0:
            return signal_audio, np.array([])
        
        # Ensure same length
        min_len = min(len(signal_audio), len(noise_audio))
        signal_audio = signal_audio[:min_len]
        noise_audio = noise_audio[:min_len]
        
        # Multi-window spectral analysis
        window_sizes = [512, 1024, 2048, 4096]
        all_shifts = []
        
        for window_size in window_sizes:
            shifts = self.windowed_spectral_analysis(signal_audio, noise_audio, window_size)
            if len(shifts) > 0:
                all_shifts.append(shifts)
        
        if len(all_shifts) == 0:
            return signal_audio, np.array([])
        
        # Combine shifts from different window sizes
        min_len = min([len(s) for s in all_shifts])
        combined_shifts = np.zeros(min_len)
        
        for shifts in all_shifts:
            combined_shifts += shifts[:min_len]
        
        combined_shifts /= len(all_shifts)  # Average
        
        # Apply enhanced noise removal
        cleaned_audio = self.enhanced_noise_removal(signal_audio, noise_audio, combined_shifts)
        
        return cleaned_audio, combined_shifts
    
    def windowed_spectral_analysis(self, signal_audio, noise_audio, window_size):
        """Perform spectral analysis with specific window size"""
        try:
            # Calculate number of windows
            n_windows = len(signal_audio) // window_size
            if n_windows == 0:
                return np.array([])
            
            shifts = np.zeros(len(signal_audio))
            
            for i in range(n_windows):
                start_idx = i * window_size
                end_idx = start_idx + window_size
                
                signal_window = signal_audio[start_idx:end_idx]
                noise_window = noise_audio[start_idx:end_idx]
                
                # Apply window function
                window = np.hanning(window_size)
                signal_windowed = signal_window * window
                noise_windowed = noise_window * window
                
                # FFT analysis
                signal_fft = fft(signal_windowed)
                noise_fft = fft(noise_windowed)
                
                # Calculate spectral features
                signal_magnitude = np.abs(signal_fft)
                noise_magnitude = np.abs(noise_fft)
                
                # Spectral subtraction with adaptive parameters
                alpha = 2.0  # Oversubtraction factor
                beta = 0.1   # Spectral floor
                
                cleaned_magnitude = signal_magnitude - alpha * noise_magnitude
                spectral_floor = beta * signal_magnitude
                cleaned_magnitude = np.maximum(cleaned_magnitude, spectral_floor)
                
                # Calculate shift for this window
                window_shift = np.mean(np.abs(signal_magnitude - cleaned_magnitude))
                
                # Assign shift to all samples in this window
                shifts[start_idx:end_idx] = window_shift
            
            return shifts
            
        except Exception as e:
            print(f"⚠️ Error in windowed analysis: {e}")
            return np.array([])
    
    def enhanced_noise_removal(self, signal_audio, noise_audio, shifts):
        """Enhanced noise removal based on shift patterns"""
        try:
            # Use multiple denoising techniques
            
            # Method 1: Adaptive spectral subtraction
            cleaned_1 = self.adaptive_spectral_subtraction(signal_audio, noise_audio, shifts)
            
            # Method 2: Wiener filtering with shift weighting
            cleaned_2 = self.shift_weighted_wiener_filter(signal_audio, noise_audio, shifts)
            
            # Method 3: Morphological filtering for impulse noise
            cleaned_3 = self.morphological_denoising(signal_audio, shifts)
            
            # Combine methods with adaptive weights based on local shift patterns
            weights = self.calculate_adaptive_weights(shifts)
            
            cleaned_combined = (weights[0] * cleaned_1 + 
                              weights[1] * cleaned_2 + 
                              weights[2] * cleaned_3)
            
            return cleaned_combined
            
        except Exception as e:
            print(f"⚠️ Error in noise removal: {e}")
            return signal_audio
    
    def adaptive_spectral_subtraction(self, signal_audio, noise_audio, shifts):
        """Spectral subtraction with shift-based adaptation"""
        if len(signal_audio) == 0 or len(noise_audio) == 0:
            return signal_audio
        
        # Use shifts to adapt subtraction parameters
        shift_percentile = np.percentile(shifts, 75)
        alpha = 1.5 + shift_percentile * 2.0  # Adaptive oversubtraction
        beta = 0.05 + shift_percentile * 0.1   # Adaptive floor
        
        # Apply windowing
        window = np.hanning(len(signal_audio))
        signal_windowed = signal_audio * window
        noise_windowed = noise_audio * window
        
        # FFT
        signal_fft = fft(signal_windowed)
        noise_fft = fft(noise_windowed)
        
        signal_magnitude = np.abs(signal_fft)
        noise_magnitude = np.abs(noise_fft)
        signal_phase = np.angle(signal_fft)
        
        # Adaptive spectral subtraction
        cleaned_magnitude = signal_magnitude - alpha * noise_magnitude
        spectral_floor = beta * signal_magnitude
        cleaned_magnitude = np.maximum(cleaned_magnitude, spectral_floor)
        
        # Reconstruct
        cleaned_fft = cleaned_magnitude * np.exp(1j * signal_phase)
        cleaned_audio = np.real(ifft(cleaned_fft))
        
        return cleaned_audio
    
    def shift_weighted_wiener_filter(self, signal_audio, noise_audio, shifts):
        """Wiener filter with shift-based weighting"""
        if len(signal_audio) == 0 or len(noise_audio) == 0:
            return signal_audio
        
        # FFT
        signal_fft = fft(signal_audio)
        noise_fft = fft(noise_audio)
        
        signal_psd = np.abs(signal_fft) ** 2
        noise_psd = np.abs(noise_fft) ** 2
        
        # Shift-weighted noise estimation
        shift_weights = shifts / (np.max(shifts) + 1e-10)
        noise_factor = 0.1 + 0.3 * np.mean(shift_weights)
        
        noise_psd_est = np.maximum(noise_factor * np.mean(noise_psd), 0.01 * signal_psd)
        
        # Wiener filter
        wiener_filter_coeff = signal_psd / (signal_psd + noise_psd_est)
        
        # Apply filter
        filtered_fft = signal_fft * wiener_filter_coeff
        filtered_audio = np.real(ifft(filtered_fft))
        
        return filtered_audio
    
    def morphological_denoising(self, signal_audio, shifts):
        """Morphological operations for impulse noise removal"""
        from scipy.ndimage import median_filter, maximum_filter, minimum_filter
        
        # Adaptive kernel size based on shift patterns
        shift_percentile = np.percentile(shifts, 90)
        kernel_size = max(3, int(5 + shift_percentile * 10))
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        
        # Apply morphological operations
        median_filtered = median_filter(signal_audio, size=kernel_size)
        
        # Combine with original based on shift patterns
        shift_norm = shifts / (np.max(shifts) + 1e-10)
        filtered_audio = signal_audio * (1 - shift_norm) + median_filtered * shift_norm
        
        return filtered_audio
    
    def calculate_adaptive_weights(self, shifts):
        """Calculate adaptive weights for combining denoising methods"""
        shift_mean = np.mean(shifts)
        shift_std = np.std(shifts)
        
        # Adapt weights based on shift statistics
        if shift_std > 0.1:  # High variation - favor spectral methods
            weights = [0.5, 0.3, 0.2]
        elif shift_mean > 0.2:  # High shifts - favor wiener
            weights = [0.3, 0.5, 0.2]
        else:  # Low shifts - balanced approach
            weights = [0.4, 0.3, 0.3]
        
        return weights
    
    def calculate_comprehensive_shifts(self, original_audio, cleaned_audio):
        """Calculate comprehensive shift metrics"""
        if len(original_audio) != len(cleaned_audio):
            min_len = min(len(original_audio), len(cleaned_audio))
            original_audio = original_audio[:min_len]
            cleaned_audio = cleaned_audio[:min_len]
        
        # Multiple shift metrics
        amplitude_shifts = np.abs(original_audio - cleaned_audio)
        energy_shifts = (original_audio ** 2) - (cleaned_audio ** 2)
        phase_shifts = self.calculate_phase_shifts(original_audio, cleaned_audio)
        
        # Combine metrics
        combined_shifts = (0.4 * amplitude_shifts + 
                          0.3 * np.abs(energy_shifts) + 
                          0.3 * phase_shifts)
        
        return combined_shifts
    
    def calculate_phase_shifts(self, original_audio, cleaned_audio):
        """Calculate phase-based shift metrics"""
        try:
            # Use short windows for local phase analysis
            window_size = 256
            n_windows = len(original_audio) // window_size
            
            phase_shifts = np.zeros(len(original_audio))
            
            for i in range(n_windows):
                start_idx = i * window_size
                end_idx = start_idx + window_size
                
                orig_window = original_audio[start_idx:end_idx]
                clean_window = cleaned_audio[start_idx:end_idx]
                
                # FFT
                orig_fft = fft(orig_window)
                clean_fft = fft(clean_window)
                
                # Phase difference
                orig_phase = np.angle(orig_fft)
                clean_phase = np.angle(clean_fft)
                
                phase_diff = np.abs(orig_phase - clean_phase)
                phase_shift = np.mean(phase_diff)
                
                phase_shifts[start_idx:end_idx] = phase_shift
            
            return phase_shifts
            
        except Exception as e:
            print(f"⚠️ Error calculating phase shifts: {e}")
            return np.zeros(len(original_audio))
    
    def detect_anomalies_with_clustering(self, shift_scores, flagged_indices, method='dbscan'):
        """Use clustering to identify anomalies based on flagged references"""
        
        print(f"🎯 Using clustering-based anomaly detection: {method}")
        
        # Prepare data
        indices = list(shift_scores.keys())
        scores = [shift_scores[idx] for idx in indices]
        
        if len(scores) == 0:
            return set()
        
        # Create feature matrix
        features = np.array(scores).reshape(-1, 1)
        
        if method == 'dbscan':
            # Use DBSCAN clustering
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Adapt eps based on flagged sample distribution
            flagged_scores = [shift_scores[idx] for idx in indices if idx in flagged_indices]
            if len(flagged_scores) > 0:
                flagged_std = np.std(flagged_scores)
                eps = max(0.1, flagged_std * 0.5)
            else:
                eps = 0.3
            
            clustering = DBSCAN(eps=eps, min_samples=5).fit(features_scaled)
            labels = clustering.labels_
            
            # Identify anomaly cluster
            unique_labels = set(labels)
            anomaly_indices = set()
            
            for label in unique_labels:
                if label == -1:  # Noise points in DBSCAN
                    cluster_indices = [indices[i] for i in range(len(indices)) if labels[i] == label]
                    anomaly_indices.update(cluster_indices)
                else:
                    # Check if this cluster contains many flagged samples
                    cluster_indices = [indices[i] for i in range(len(indices)) if labels[i] == label]
                    flagged_in_cluster = sum(1 for idx in cluster_indices if idx in flagged_indices)
                    flagged_ratio = flagged_in_cluster / len(cluster_indices) if len(cluster_indices) > 0 else 0
                    
                    # If cluster has high ratio of flagged samples, mark as anomalous
                    if flagged_ratio > 0.3:  # Threshold for flagged ratio
                        anomaly_indices.update(cluster_indices)
        
        elif method == 'percentile':
            # Use percentile-based approach guided by flagged samples
            flagged_scores = [shift_scores[idx] for idx in indices if idx in flagged_indices]
            
            if len(flagged_scores) > 0:
                # Use flagged samples to determine threshold
                flagged_percentile = np.percentile(flagged_scores, 10)  # Lower bound of flagged scores
                threshold = max(flagged_percentile, np.percentile(scores, 85))
            else:
                threshold = np.percentile(scores, 90)
            
            anomaly_indices = set()
            for idx, score in shift_scores.items():
                if score >= threshold:
                    anomaly_indices.add(idx)
        
        elif method == 'flagged_guided':
            # Directly use flagged samples as anomalies plus high shift scores
            anomaly_indices = set(flagged_indices)
            
            # Add samples with exceptionally high shift scores
            if len(scores) > 0:
                high_threshold = np.mean(scores) + 2 * np.std(scores)
                for idx, score in shift_scores.items():
                    if score >= high_threshold:
                        anomaly_indices.add(idx)
        
        print(f"🚨 Detected {len(anomaly_indices)} anomalous samples")
        print(f"📊 Flagged samples included: {len(anomaly_indices.intersection(flagged_indices))}")
        
        return anomaly_indices
    
    def process_all_columns(self, csv_file, flagged_file, detection_method='flagged_guided'):
        """Process all audio columns and detect anomalies"""
        
        print(f"🎵 Processing all audio columns...")
        
        # Load flagged indices
        flagged_indices = self.load_flagged_indices(flagged_file)
        
        # Load original data to get column info
        try:
            df = pd.read_csv(csv_file)
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            print(f"🔢 Processing {len(numeric_columns)} numeric columns")
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            return set()
        
        # Collect shift data for all columns
        all_sample_shifts = defaultdict(list)
        processing_results = {}
        
        # Process each column
        for column in numeric_columns:
            print(f"\n🎵 Processing column: {column}")
            
            # File paths
            standard_file = os.path.join(self.output_dir, f"{column}_standard.wav")
            anomalies_file = os.path.join(self.output_dir, f"{column}_anomalies_only.wav")
            
            # Check if files exist
            if not os.path.exists(standard_file):
                print(f"⚠️ Standard audio file not found for {column}")
                continue
            
            if not os.path.exists(anomalies_file):
                print(f"⚠️ Anomalies audio file not found for {column}")
                continue
            
            # Read audio files
            standard_audio = self.read_wav_file(standard_file)
            anomalies_audio = self.read_wav_file(anomalies_file)
            
            if len(standard_audio) == 0:
                continue
            
            # Apply advanced spectral analysis
            cleaned_audio, raw_shifts = self.advanced_spectral_analysis(standard_audio, anomalies_audio)
            
            # Save cleaned audio
            cleaned_filename = os.path.join(self.output_dir, f"{column}_enhanced_cleaned.wav")
            self.write_wav_file(cleaned_audio, cleaned_filename)
            
            # Calculate comprehensive shifts
            comprehensive_shifts = self.calculate_comprehensive_shifts(standard_audio, cleaned_audio)
            
            # Map audio shifts to data indices
            data_shifts = self.map_audio_shifts_to_data_indices(comprehensive_shifts, len(df))
            
            # Store results
            processing_results[column] = {
                'shifts': comprehensive_shifts,
                'data_shifts': data_shifts,
                'standard_audio': standard_audio,
                'cleaned_audio': cleaned_audio
            }
            
            # Store shifts for each row
            for i, shift in enumerate(data_shifts):
                all_sample_shifts[i].append(shift)
        
        # Calculate combined shift score for each row
        shift_scores = {}
        for row_idx, shifts_list in all_sample_shifts.items():
            if len(shifts_list) > 0:
                # Use maximum shift across columns (most sensitive to anomalies)
                combined_shift = np.max(shifts_list)
                shift_scores[row_idx] = combined_shift
        
        if len(shift_scores) == 0:
            print("⚠️ No shift data available")
            return set()
        
        # Detect anomalies using chosen method
        anomalous_indices = self.detect_anomalies_with_clustering(
            shift_scores, flagged_indices, method=detection_method
        )
        
        # Save comprehensive analysis results
        self.save_analysis_results(shift_scores, anomalous_indices, flagged_indices, 
                                 processing_results, detection_method)
        
        # Create visualizations
        self.create_comprehensive_visualizations(shift_scores, anomalous_indices, 
                                               flagged_indices, detection_method)
        
        return anomalous_indices
    
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
            
            # Average shifts in a window around the mapped position
            window_size = max(1, audio_length // original_data_length)
            start_idx = max(0, audio_idx - window_size // 2)
            end_idx = min(audio_length, audio_idx + window_size // 2 + 1)
            
            data_shifts[i] = np.mean(shifts[start_idx:end_idx])
        
        return data_shifts
    
    def save_analysis_results(self, shift_scores, anomalous_indices, flagged_indices, 
                            processing_results, method):
        """Save comprehensive analysis results"""
        
        analysis_data = {
            'detection_method': method,
            'total_samples': len(shift_scores),
            'total_anomalies': len(anomalous_indices),
            'total_flagged': len(flagged_indices),
            'flagged_detected': len(anomalous_indices.intersection(flagged_indices)),
            'shift_scores': {str(k): float(v) for k, v in shift_scores.items()},
            'anomalous_indices': sorted(list(anomalous_indices)),
            'flagged_indices': sorted(list(flagged_indices)),
            'processing_stats': {
                'columns_processed': len(processing_results),
                'mean_shift': float(np.mean(list(shift_scores.values()))),
                'std_shift': float(np.std(list(shift_scores.values()))),
                'max_shift': float(np.max(list(shift_scores.values()))),
                'min_shift': float(np.min(list(shift_scores.values())))
            }
        }
        
        analysis_file = os.path.join(self.output_dir, "enhanced_audio_analysis.json")
        try:
            with open(analysis_file, 'w') as f:
                json.dump(analysis_data, f, indent=2)
            print(f"📝 Enhanced analysis saved to: {analysis_file}")
        except Exception as e:
            print(f"⚠️ Could not save analysis: {e}")
    
    def create_comprehensive_visualizations(self, shift_scores, anomalous_indices, 
                                          flagged_indices, method):
        """Create comprehensive visualizations"""
        
        try:
            # Main analysis plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
            
            # Prepare data
            indices = sorted(shift_scores.keys())
            scores = [shift_scores[idx] for idx in indices]
            
            # Plot 1: Shift scores with anomalies highlighted
            ax1.scatter(indices, scores, alpha=0.5, s=20, color='blue', label='Normal samples')
            
            # Highlight flagged samples
            flagged_in_data = [idx for idx in indices if idx in flagged_indices]
            flagged_scores = [shift_scores[idx] for idx in flagged_in_data]
            if len(flagged_in_data) > 0:
                ax1.scatter(flagged_in_data, flagged_scores, color='orange', s=30, 
                           alpha=0.8, label=f'Originally Flagged ({len(flagged_in_data)})')
            
            # Highlight detected anomalies
            anomaly_in_data = [idx for idx in indices if idx in anomalous_indices]
            anomaly_scores = [shift_scores[idx] for idx in anomaly_in_data]
            if len(anomaly_in_data) > 0:
                ax1.scatter(anomaly_in_data, anomaly_scores, color='red', s=25, 
                           alpha=0.8, label=f'Detected Anomalies ({len(anomaly_in_data)})')
            
            ax1.set_xlabel('Row Index')
            ax1.set_ylabel('Audio Shift Score')
            ax1.set_title(f'Enhanced Audio-Based Anomaly Detection\nMethod: {method}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Distribution comparison
            ax2.hist(scores, bins=50, alpha=0.7, color='blue', label='All samples', density=True)
            
            if len(flagged_scores) > 0:
                ax2.hist(flagged_scores, bins=20, alpha=0.7, color='orange', 
                        label='Flagged samples', density=True)
            
            if len(anomaly_scores) > 0:
                ax2.hist(anomaly_scores, bins=20, alpha=0.7, color='red', 
                        label='Detected anomalies', density=True)
            
            ax2.set_xlabel('Audio Shift Score')
            ax2.set_ylabel('Density')
            ax2.set_title('Distribution of Audio Shift Scores')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save main plot
            main_plot_file = os.path.join(self.output_dir, f"enhanced_anomaly_detection_{method}.png")
            plt.savefig(main_plot_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"📈 Enhanced visualization saved to: {main_plot_file}")
            
            # Create method comparison if multiple methods available
            self.create_method_comparison(shift_scores, flagged_indices)
            
        except Exception as e:
            print(f"⚠️ Could not create visualizations: {e}")
    
    def create_method_comparison(self, shift_scores, flagged_indices):
        """Compare different detection methods"""
        
        try:
            methods = ['dbscan', 'percentile', 'flagged_guided']
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            indices = sorted(shift_scores.keys())
            scores = [shift_scores[idx] for idx in indices]
            
            for i, method in enumerate(methods):
                ax = axes[i]
                
                # Detect anomalies with this method
                anomalies = self.detect_anomalies_with_clustering(shift_scores, flagged_indices, method)
                
                # Plot
                ax.scatter(indices, scores, alpha=0.4, s=15, color='blue', label='Normal')
                
                # Highlight anomalies
                anomaly_indices = [idx for idx in indices if idx in anomalies]
                anomaly_scores = [shift_scores[idx] for idx in anomaly_indices]
                if len(anomaly_indices) > 0:
                    ax.scatter(anomaly_indices, anomaly_scores, color='red', s=20, 
                             alpha=0.8, label=f'Anomalies ({len(anomaly_indices)})')
                
                overlap = len(anomalies.intersection(flagged_indices))
                ax.set_title(f'{method.title()} Method\nDetected: {len(anomalies)}, '
                           f'Overlap with Flagged: {overlap}')
                ax.set_xlabel('Row Index')
                ax.set_ylabel('Shift Score')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            comparison_file = os.path.join(self.output_dir, "method_comparison.png")
            plt.savefig(comparison_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Method comparison saved to: {comparison_file}")
            
        except Exception as e:
            print(f"⚠️ Could not create method comparison: {e}")
    
    def create_clean_csv(self, csv_file, anomalous_indices, output_file="cleaned_data_enhanced.csv"):
        """Create clean CSV by removing detected anomalies"""
        
        print(f"🧹 Creating enhanced clean CSV...")
        
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
            
            print(f"✅ Enhanced clean CSV created: {output_file}")
            print(f"📊 Original rows: {original_count}")
            print(f"📊 Removed rows: {removed_count}")
            print(f"📊 Clean rows: {clean_count}")
            print(f"📊 Removal rate: {(removed_count/original_count)*100:.2f}%")
            
            # Create detailed summary
            summary_file = output_file.replace('.csv', '_summary.txt')
            with open(summary_file, 'w') as f:
                f.write("Enhanced Audio-Based Anomaly Detection Summary\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Input file: {csv_file}\n")
                f.write(f"Output file: {output_file}\n")
                f.write(f"Detection method: Enhanced audio processing with clustering\n")
                f.write(f"Original rows: {original_count}\n")
                f.write(f"Removed rows: {removed_count}\n")
                f.write(f"Clean rows: {clean_count}\n")
                f.write(f"Removal rate: {(removed_count/original_count)*100:.2f}%\n\n")
                f.write("Removed row indices:\n")
                for idx in sorted(list(anomalous_indices)):
                    f.write(f"{idx}\n")
            
            print(f"📝 Summary saved to: {summary_file}")
            return df_clean
            
        except Exception as e:
            print(f"❌ Error creating clean CSV: {e}")
            return None


def main():
    """Main function for enhanced audio-based anomaly detection"""
    
    # Configuration
    CSV_FILE = "output/cleaned_firstpass.csv"
    FLAGGED_FILE = "flagged_enhanced.txt"
    AUDIO_DIR = "audio_analysis"
    OUTPUT_CSV = "cleaned_data_final_enhanced.csv"
    
    # Detection method: 'dbscan', 'percentile', 'flagged_guided'
    DETECTION_METHOD = 'percentile'
    
    print("🎵 Enhanced Audio-Based Anomaly Detection System")
    print("=" * 55)
    print(f"🎯 Detection method: {DETECTION_METHOD}")
    print(f"📁 Audio directory: {AUDIO_DIR}")
    print(f"📋 Flagged file: {FLAGGED_FILE}")
    
    # Initialize processor
    processor = AudioAnomalyProcessor(output_dir=AUDIO_DIR)
    
    # Check if audio files exist
    if not os.path.exists(AUDIO_DIR):
        print(f"❌ Audio directory not found: {AUDIO_DIR}")
        print("Please run the audio converter script first!")
        return
    
    # Check if flagged file exists
    if not os.path.exists(FLAGGED_FILE):
        print(f"❌ Flagged file not found: {FLAGGED_FILE}")
        print("Please ensure the flagged indices file exists!")
        return
    
    # Process all columns and detect anomalies
    anomalous_indices = processor.process_all_columns(
        csv_file=CSV_FILE,
        flagged_file=FLAGGED_FILE,
        detection_method=DETECTION_METHOD
    )
    
    if len(anomalous_indices) == 0:
        print("⚠️ No anomalies detected through enhanced audio analysis")
        return
    
    # Create clean CSV
    cleaned_df = processor.create_clean_csv(
        csv_file=CSV_FILE,
        anomalous_indices=anomalous_indices,
        output_file=OUTPUT_CSV
    )
    
    if cleaned_df is not None:
        print(f"\n🎉 Enhanced audio processing completed successfully!")
        print(f"📁 Clean data saved to: {OUTPUT_CSV}")
        print(f"🔊 Enhanced analysis files in: {AUDIO_DIR}")
        print(f"📈 Visualizations show detected anomalies vs flagged samples")
        
        # Print performance metrics
        flagged_indices = processor.load_flagged_indices(FLAGGED_FILE)
        overlap = len(anomalous_indices.intersection(flagged_indices))
        print(f"\n📊 Performance Metrics:")
        print(f"   • Total anomalies detected: {len(anomalous_indices)}")
        print(f"   • Originally flagged samples: {len(flagged_indices)}")
        print(f"   • Overlap (flagged samples detected): {overlap}")
        print(f"   • Detection rate: {(overlap/len(flagged_indices)*100):.1f}%")


# Alternative detection methods for different use cases
class AudioAnomalyMethods:
    """Alternative detection methods for different scenarios"""
    
    @staticmethod
    def energy_based_detection(shift_scores, flagged_indices, energy_threshold=0.8):
        """Energy-based anomaly detection"""
        scores = list(shift_scores.values())
        threshold = np.percentile(scores, energy_threshold * 100)
        
        anomalies = set()
        for idx, score in shift_scores.items():
            if score >= threshold:
                anomalies.add(idx)
        
        # Always include flagged samples
        anomalies.update(flagged_indices)
        return anomalies
    
    @staticmethod
    def statistical_outlier_detection(shift_scores, flagged_indices, z_threshold=2.5):
        """Statistical outlier detection using Z-scores"""
        scores = list(shift_scores.values())
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        anomalies = set()
        for idx, score in shift_scores.items():
            z_score = abs(score - mean_score) / (std_score + 1e-10)
            if z_score >= z_threshold:
                anomalies.add(idx)
        
        # Always include flagged samples
        anomalies.update(flagged_indices)
        return anomalies
    
    @staticmethod
    def isolation_forest_detection(shift_scores, flagged_indices, contamination=0.1):
        """Use Isolation Forest for anomaly detection"""
        try:
            from sklearn.ensemble import IsolationForest
            
            indices = list(shift_scores.keys())
            scores = [[shift_scores[idx]] for idx in indices]
            
            # Fit Isolation Forest
            iso_forest = IsolationForest(contamination=contamination, random_state=42)
            predictions = iso_forest.fit_predict(scores)
            
            # Get anomalies (predictions == -1)
            anomalies = set()
            for i, pred in enumerate(predictions):
                if pred == -1:
                    anomalies.add(indices[i])
            
            # Always include flagged samples
            anomalies.update(flagged_indices)
            return anomalies
            
        except ImportError:
            print("⚠️ sklearn not available, falling back to statistical method")
            return AudioAnomalyMethods.statistical_outlier_detection(
                shift_scores, flagged_indices
            )


# Usage example with different methods
def run_method_comparison():
    """Run comparison of different detection methods"""
    
    CSV_FILE = "output/cleaned_firstpass.csv"
    FLAGGED_FILE = "flagged_enhanced.txt"
    AUDIO_DIR = "audio_analysis"
    
    processor = AudioAnomalyProcessor(output_dir=AUDIO_DIR)
    
    # Load data
    flagged_indices = processor.load_flagged_indices(FLAGGED_FILE)
    
    # This would need to be implemented to load pre-computed shift scores
    # For now, this is a template for how you could compare methods
    shift_scores = {}  # Load from previous analysis
    
    methods = {
        'dbscan': lambda: processor.detect_anomalies_with_clustering(
            shift_scores, flagged_indices, 'dbscan'
        ),
        'percentile': lambda: processor.detect_anomalies_with_clustering(
            shift_scores, flagged_indices, 'percentile'
        ),
        'flagged_guided': lambda: processor.detect_anomalies_with_clustering(
            shift_scores, flagged_indices, 'flagged_guided'
        ),
        'energy_based': lambda: AudioAnomalyMethods.energy_based_detection(
            shift_scores, flagged_indices
        ),
        'statistical': lambda: AudioAnomalyMethods.statistical_outlier_detection(
            shift_scores, flagged_indices
        ),
        'isolation_forest': lambda: AudioAnomalyMethods.isolation_forest_detection(
            shift_scores, flagged_indices
        )
    }
    
    print("🔬 Comparing detection methods:")
    for method_name, method_func in methods.items():
        try:
            anomalies = method_func()
            overlap = len(anomalies.intersection(flagged_indices))
            detection_rate = (overlap / len(flagged_indices) * 100) if len(flagged_indices) > 0 else 0
            
            print(f"   • {method_name}: {len(anomalies)} detected, "
                  f"{overlap} overlap, {detection_rate:.1f}% detection rate")
        except Exception as e:
            print(f"   • {method_name}: Failed - {e}")


if __name__ == "__main__":
    main()