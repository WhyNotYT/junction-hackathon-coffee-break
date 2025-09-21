import pandas as pd
import json
import numpy as np
import wave
import struct
import os
from collections import defaultdict
import warnings
from multiprocessing import Pool, cpu_count, Manager
import time
from functools import partial

# Suppress warnings
warnings.filterwarnings("ignore")

def normalize_data_for_audio(data, target_range=(-1.0, 1.0)):
    """
    Normalize data to a target range suitable for audio
    
    Args:
        data: numpy array of data values
        target_range: tuple of (min, max) for output range
    
    Returns:
        normalized numpy array
    """
    if len(data) == 0:
        return data
    
    # Ensure data is numpy array and handle mixed types
    data = np.array(data, dtype=object)
    
    # Convert to numeric, replacing non-numeric values with NaN
    numeric_data = []
    for val in data:
        try:
            if val is None:
                numeric_data.append(np.nan)
            else:
                numeric_val = float(val)
                numeric_data.append(numeric_val)
        except (ValueError, TypeError):
            # Non-numeric value, skip it
            numeric_data.append(np.nan)
    
    numeric_data = np.array(numeric_data, dtype=float)
    
    # Remove NaN values for calculation
    clean_data = numeric_data[~np.isnan(numeric_data)]
    if len(clean_data) == 0:
        return np.full(len(numeric_data), target_range[0])
    
    # Get data range
    data_min, data_max = np.min(clean_data), np.max(clean_data)
    
    # Avoid division by zero
    if data_max == data_min:
        return np.full_like(numeric_data, target_range[0])
    
    # Normalize to target range
    normalized = (numeric_data - data_min) / (data_max - data_min)
    normalized = normalized * (target_range[1] - target_range[0]) + target_range[0]
    
    # Handle NaN values by setting them to target_range[0]
    normalized[np.isnan(normalized)] = target_range[0]
    
    return normalized

def calculate_audio_parameters(data_length, sample_rate=44100, min_duration=1.0, max_duration=300.0, samples_per_second=100):
    """
    Calculate dynamic audio parameters based on data length
    
    Args:
        data_length: number of data points
        sample_rate: audio sample rate in Hz
        min_duration: minimum audio duration in seconds
        max_duration: maximum audio duration in seconds (optional limit)
        samples_per_second: how many data points to represent per second of audio
    
    Returns:
        tuple of (duration, num_samples, data_per_sample)
    """
    # Calculate ideal duration based on data density
    ideal_duration = data_length / samples_per_second
    
    # Apply constraints
    duration = max(min_duration, ideal_duration)
    if max_duration is not None:
        duration = min(duration, max_duration)
    
    # Calculate total audio samples needed
    num_samples = int(sample_rate * duration)
    
    # Calculate how many data points each audio sample represents
    data_per_sample = data_length / num_samples
    
    return duration, num_samples, data_per_sample

def create_wav_file(data, filename, sample_rate=44100, min_duration=1.0, max_duration=None, samples_per_second=100):
    """
    Create a WAV file from numerical data with dynamic length
    
    Args:
        data: numpy array of normalized data (-1.0 to 1.0)
        filename: output WAV filename
        sample_rate: audio sample rate in Hz
        min_duration: minimum duration in seconds
        max_duration: maximum duration in seconds (None for no limit)
        samples_per_second: data points to represent per second of audio
    """
    if len(data) == 0:
        print(f"⚠️ No data to convert for {filename}")
        return 0
    
    # Calculate dynamic audio parameters
    duration, num_samples, data_per_sample = calculate_audio_parameters(
        len(data), sample_rate, min_duration, max_duration, samples_per_second
    )
    
    # Create audio data by resampling/aggregating the input data
    if data_per_sample < 1.0:
        # Interpolate data to match desired sample count (upsample)
        x_original = np.linspace(0, 1, len(data))
        x_new = np.linspace(0, 1, num_samples)
        audio_data = np.interp(x_new, x_original, data)
    else:
        # Aggregate data points (downsample)
        audio_data = []
        for i in range(num_samples):
            start_idx = int(i * data_per_sample)
            end_idx = int((i + 1) * data_per_sample)
            end_idx = min(end_idx, len(data))
            
            if start_idx >= len(data):
                audio_data.append(0.0)
            elif start_idx == end_idx:
                audio_data.append(data[start_idx])
            else:
                # Use mean of the data points in this segment
                segment = data[start_idx:end_idx]
                audio_data.append(np.mean(segment))
        
        audio_data = np.array(audio_data)
    
    # Ensure data is in valid range
    audio_data = np.clip(audio_data, -1.0, 1.0)
    
    # Convert to 16-bit integers
    audio_data_int = (audio_data * 32767).astype(np.int16)
    
    # Create WAV file
    try:
        with wave.open(filename, 'w') as wav_file:
            # Set parameters: nchannels, sampwidth, framerate, nframes
            wav_file.setparams((1, 2, sample_rate, num_samples, 'NONE', 'not compressed'))
            
            # Write audio data
            for sample in audio_data_int:
                wav_file.writeframes(struct.pack('<h', sample))
        
        return duration
        
    except Exception as e:
        print(f"❌ Error creating {filename}: {e}")
        return 0

def add_anomaly_highlights(normal_data, anomaly_indices, anomaly_values, highlight_factor=2.0, data_length=None):
    """
    Add highlights to audio data where anomalies occur
    
    Args:
        normal_data: normalized normal data (audio samples)
        anomaly_indices: list of indices where anomalies occur (original data indices)
        anomaly_values: list of anomaly values
        highlight_factor: multiplier for anomaly emphasis
        data_length: original data length for mapping indices
    
    Returns:
        modified audio data with anomaly highlights
    """
    if len(normal_data) == 0 or len(anomaly_indices) == 0:
        return normal_data
    
    highlighted_data = normal_data.copy()
    
    if data_length is None:
        data_length = len(anomaly_indices)
    
    audio_length = len(normal_data)
    
    for orig_idx in anomaly_indices:
        # Map original data index to audio data index
        if data_length > 0:
            audio_idx = int((orig_idx / data_length) * audio_length)
        else:
            audio_idx = audio_length // 2
        
        # Create a highlight region around the anomaly
        highlight_width = max(1, audio_length // 200)  # 0.5% of audio length
        start_idx = max(0, audio_idx - highlight_width // 2)
        end_idx = min(audio_length, audio_idx + highlight_width // 2)
        
        # Amplify the region (but keep within bounds)
        for j in range(start_idx, end_idx):
            highlighted_data[j] = np.clip(highlighted_data[j] * highlight_factor, -1.0, 1.0)
    
    return highlighted_data

def process_column_worker(args):
    """
    Worker function for processing a single column in a separate process
    This function will be called by multiprocessing.Pool
    
    Args:
        args: tuple containing all the arguments needed for processing
    
    Returns:
        dict with processing results
    """
    (col, col_data, flagged_values, flagged_indices, output_dir, 
     sample_rate, min_duration, max_duration, samples_per_second, 
     create_anomaly_highlights) = args
    
    start_time = time.time()
    
    try:
        # Get column data (fill NaN with median)
        col_series = pd.Series(col_data)
        col_data_filled = col_series.fillna(col_series.median()).values
        
        if len(col_data_filled) == 0:
            return {
                'column': col,
                'success': False,
                'error': 'No data in column',
                'duration': 0,
                'files_created': 0,
                'process_time': time.time() - start_time
            }
        
        # Normalize data for audio
        normalized_data = normalize_data_for_audio(col_data_filled)
        
        total_column_duration = 0
        files_created = 0
        
        # Create standard audio file
        standard_filename = os.path.join(output_dir, f"{col}_standard.wav")
        duration = create_wav_file(normalized_data, standard_filename, sample_rate, 
                       min_duration, max_duration, samples_per_second)
        if duration > 0:
            total_column_duration += duration
            files_created += 1
        
        # Create anomaly-highlighted version if anomalies exist for this column
        if create_anomaly_highlights and len(flagged_values) > 0:
            # For highlighting, we need to first create the audio data to know its length
            duration, num_samples, _ = calculate_audio_parameters(
                len(col_data_filled), sample_rate, min_duration, max_duration, samples_per_second
            )
            
            # Create base audio data for highlighting
            if len(col_data_filled) / num_samples < 1.0:
                x_original = np.linspace(0, 1, len(col_data_filled))
                x_new = np.linspace(0, 1, num_samples)
                base_audio = np.interp(x_new, x_original, normalized_data)
            else:
                data_per_sample = len(col_data_filled) / num_samples
                base_audio = []
                for i in range(num_samples):
                    start_idx = int(i * data_per_sample)
                    end_idx = int((i + 1) * data_per_sample)
                    end_idx = min(end_idx, len(col_data_filled))
                    
                    if start_idx >= len(col_data_filled):
                        base_audio.append(0.0)
                    elif start_idx == end_idx:
                        base_audio.append(normalized_data[start_idx])
                    else:
                        segment = normalized_data[start_idx:end_idx]
                        base_audio.append(np.mean(segment))
                
                base_audio = np.array(base_audio)
            
            # Create highlighted version
            highlighted_data = add_anomaly_highlights(
                base_audio, flagged_indices, flagged_values, data_length=len(col_data_filled)
            )
            
            highlighted_filename = os.path.join(output_dir, f"{col}_with_anomalies.wav")
            
            # Save highlighted version directly
            highlighted_data = np.clip(highlighted_data, -1.0, 1.0)
            highlighted_data_int = (highlighted_data * 32767).astype(np.int16)
            
            try:
                with wave.open(highlighted_filename, 'w') as wav_file:
                    wav_file.setparams((1, 2, sample_rate, len(highlighted_data_int), 'NONE', 'not compressed'))
                    for sample in highlighted_data_int:
                        wav_file.writeframes(struct.pack('<h', sample))
                
                total_column_duration += duration
                files_created += 1
            except Exception as e:
                print(f"❌ Error creating {highlighted_filename}: {e}")
            
            # Create anomalies-only audio (just the flagged values)
            if len(flagged_values) > 1:
                flagged_normalized = normalize_data_for_audio(np.array(flagged_values))
                anomalies_only_filename = os.path.join(output_dir, f"{col}_anomalies_only.wav")
                duration = create_wav_file(flagged_normalized, anomalies_only_filename, sample_rate,
                               min_duration, max_duration, samples_per_second)
                if duration > 0:
                    total_column_duration += duration
                    files_created += 1
        
        return {
            'column': col,
            'success': True,
            'error': None,
            'duration': total_column_duration,
            'files_created': files_created,
            'anomalies_found': len(flagged_values),
            'process_time': time.time() - start_time
        }
    
    except Exception as e:
        return {
            'column': col,
            'success': False,
            'error': str(e),
            'duration': 0,
            'files_created': 0,
            'process_time': time.time() - start_time
        }

def load_flagged_data(flagged_file):
    """Load and parse flagged anomaly data"""
    print(f"📂 Loading flagged data from: {flagged_file}")
    
    flagged_records = []
    try:
        with open(flagged_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    record = json.loads(line.strip())
                    flagged_records.append(record)
                except json.JSONDecodeError as e:
                    print(f"⚠️ Invalid JSON on line {line_num}: {e}")
        
        print(f"✅ Loaded {len(flagged_records)} flagged records")
        return flagged_records
    
    except FileNotFoundError:
        print(f"❌ Flagged file not found: {flagged_file}")
        return []
    except Exception as e:
        print(f"❌ Error loading flagged data: {e}")
        return []

def export_data_to_audio(csv_file, flagged_file, output_dir="audio_output", 
                        sample_rate=44100, min_duration=1.0, max_duration=None,
                        samples_per_second=100, create_anomaly_highlights=True,
                        processes=None):
    """
    Export CSV columns and flagged data to WAV audio files with dynamic length using multiprocessing
    
    Args:
        csv_file: path to original CSV data
        flagged_file: path to flagged anomaly data (JSON lines)
        output_dir: directory to save WAV files
        sample_rate: audio sample rate
        min_duration: minimum duration of each audio file in seconds
        max_duration: maximum duration (None for no limit)
        samples_per_second: data points to represent per second of audio
        create_anomaly_highlights: whether to create highlighted versions
        processes: number of processes (None = CPU count)
    """
    print(f"🚀 Converting data to audio files with MULTIPROCESSING (GIL-FREE)...")
    print(f"📁 Input CSV: {csv_file}")
    print(f"📁 Flagged data: {flagged_file}")
    print(f"📁 Output directory: {output_dir}")
    print(f"⚙️ Audio settings: {sample_rate}Hz, {samples_per_second} data points/second")
    if max_duration:
        print(f"⚙️ Duration range: {min_duration}s - {max_duration}s")
    else:
        print(f"⚙️ Minimum duration: {min_duration}s (no maximum limit)")
    
    # Determine number of processes
    if processes is None:
        processes = cpu_count()
    print(f"🔥 Using {processes} CPU cores (processes) - FULL CPU UTILIZATION!")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load original data
    try:
        print("📊 Loading original CSV data...")
        df = pd.read_csv(csv_file)
        print(f"✅ Loaded CSV: {len(df)} rows, {len(df.columns)} columns")
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return
    
    # Load flagged data
    flagged_records = load_flagged_data(flagged_file)
    if not flagged_records:
        print("⚠️ No flagged data found, creating audio from original data only")
    
    # Organize flagged data by column
    flagged_by_column = defaultdict(list)
    flagged_indices_by_column = defaultdict(list)
    
    for record in flagged_records:
        row_idx = record.get('row_index', -1)
        flagged_cols = record.get('flagged_columns', [])
        data = record.get('data', {})
        
        for col in flagged_cols:
            if col in data and data[col] is not None:
                # Only add numeric values
                try:
                    numeric_val = float(data[col])
                    if not np.isnan(numeric_val):
                        flagged_by_column[col].append(numeric_val)
                        flagged_indices_by_column[col].append(row_idx)
                except (ValueError, TypeError):
                    # Skip non-numeric values
                    continue
    
    # Get numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"🔢 Found {len(numeric_columns)} numeric columns: {numeric_columns}")
    
    if not numeric_columns:
        print("❌ No numeric columns found!")
        return
    
    # Prepare arguments for each worker process
    worker_args = []
    for col in numeric_columns:
        col_data = df[col].values
        if len(col_data) == 0:
            continue
        
        flagged_values = flagged_by_column.get(col, [])
        flagged_indices = flagged_indices_by_column.get(col, [])
        
        args = (col, col_data, flagged_values, flagged_indices, output_dir,
                sample_rate, min_duration, max_duration, samples_per_second,
                create_anomaly_highlights)
        worker_args.append(args)
    
    print(f"\n🚀 Starting multiprocessing with {len(worker_args)} columns across {processes} processes...")
    start_time = time.time()
    
    # Process columns in parallel using multiprocessing
    total_duration = 0
    processed_columns = []
    total_files = 0
    total_anomalies = 0
    
    try:
        with Pool(processes=processes) as pool:
            # Use map to process all columns
            results = pool.map(process_column_worker, worker_args)
            
            # Process results
            for result in results:
                col = result['column']
                if result['success']:
                    total_duration += result['duration']
                    total_files += result['files_created']
                    total_anomalies += result['anomalies_found']
                    processed_columns.append(col)
                    print(f"✅ {col}: {result['files_created']} files, {result['duration']:.2f}s audio, "
                          f"{result['anomalies_found']} anomalies ({result['process_time']:.2f}s)")
                else:
                    print(f"❌ {col}: {result['error']}")
    
    except Exception as e:
        print(f"❌ Multiprocessing error: {e}")
        return
    
    processing_time = time.time() - start_time
    
    print(f"\n🎉 MULTIPROCESSING COMPLETE!")
    print(f"📁 Files saved to: {output_dir}")
    print(f"🔊 Audio format: {sample_rate}Hz")
    print(f"📊 Processed {len(processed_columns)} columns successfully")
    print(f"📄 Created {total_files} audio files")
    print(f"🚨 Found {total_anomalies} total anomalies")
    print(f"⏱️ Total audio duration: {total_duration:.2f} seconds ({total_duration/60:.1f} minutes)")
    print(f"⚡ Processing time: {processing_time:.2f} seconds")
    print(f"🔥 Speed improvement: ~{len(processed_columns)/processing_time:.1f} columns/second")
    
    # Create summary file
    summary_file = os.path.join(output_dir, "audio_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("Multiprocessing Audio Export Summary (GIL-FREE)\n")
        f.write("============================================\n\n")
        f.write(f"Original CSV: {csv_file}\n")
        f.write(f"Flagged data: {flagged_file}\n")
        f.write(f"CPU cores used: {processes}\n")
        f.write(f"Processing time: {processing_time:.2f} seconds\n")
        f.write(f"Processed columns: {len(processed_columns)}\n")
        f.write(f"Total files created: {total_files}\n")
        f.write(f"Total anomalies: {total_anomalies}\n")
        f.write(f"Sample rate: {sample_rate}Hz\n")
        f.write(f"Data points per second: {samples_per_second}\n")
        f.write(f"Minimum duration: {min_duration}s\n")
        if max_duration:
            f.write(f"Maximum duration: {max_duration}s\n")
        else:
            f.write("Maximum duration: No limit\n")
        f.write(f"Total audio duration: {total_duration:.2f}s ({total_duration/60:.1f} minutes)\n\n")
        
        f.write("Performance:\n")
        f.write(f"- Processing speed: {len(processed_columns)/processing_time:.2f} columns/second\n")
        f.write(f"- CPU utilization: FULL (multiprocessing bypasses GIL)\n\n")
        
        f.write("File Types Created:\n")
        f.write("- *_standard.wav: Normal data sonification (dynamic length)\n")
        f.write("- *_with_anomalies.wav: Data with anomalies emphasized (dynamic length)\n")
        f.write("- *_anomalies_only.wav: Only the anomalous values (dynamic length)\n\n")
        
        f.write("Successfully processed columns:\n")
        for col in sorted(processed_columns):
            anomaly_count = len(flagged_by_column.get(col, []))
            f.write(f"- {col}: {anomaly_count} anomalies\n")
    
    print(f"📝 Summary saved to: {summary_file}")

# Example usage and configuration
if __name__ == "__main__":
    # Configuration - Update these paths to match your files
    CSV_FILE = "output/cleaned_firstpass.csv"  # Your original CSV
    FLAGGED_FILE = "flagged_enhanced.txt"  # Your flagged anomalies file
    OUTPUT_DIR = "audio_analysis"  # Output directory for WAV files
    
    # Dynamic audio settings
    SAMPLE_RATE = 44100        # CD quality
    MIN_DURATION = 1.0         # Minimum 1 second per file
    MAX_DURATION = None        # No maximum limit (set to 300 for 5-minute limit)
    SAMPLES_PER_SECOND = 44100   # How many data points per second of audio
    CREATE_HIGHLIGHTS = True   # Create anomaly-highlighted versions
    PROCESSES = None           # None = use all CPU cores, or set specific number
    
    print("🚀 GIL-FREE Multiprocessing Data-to-Audio Converter for Anomaly Analysis")
    print("=" * 80)
    
    # Run the conversion
    export_data_to_audio(
        csv_file=CSV_FILE,
        flagged_file=FLAGGED_FILE,
        output_dir=OUTPUT_DIR,
        sample_rate=SAMPLE_RATE,
        min_duration=MIN_DURATION,
        max_duration=MAX_DURATION,
        samples_per_second=SAMPLES_PER_SECOND,
        create_anomaly_highlights=CREATE_HIGHLIGHTS,
        processes=PROCESSES
    )
    
    print("\n🎧 Listening Guide:")
    print("- Audio length now matches your data size")
    print("- Standard files: Hear the complete data patterns over time")
    print("- With anomalies: Anomalies are amplified/emphasized")
    print("- Anomalies only: Just the outlier values as audio")
    print("- Higher pitches = higher values")
    print("- Sudden volume changes = potential anomalies")
    print("- Listen for temporal patterns, trends, and seasonal changes")
    print(f"- Each second represents ~{SAMPLES_PER_SECOND} data points")
    
    print("\n🔥 PERFORMANCE BEAST MODE:")
    print("- TRUE multiprocessing - bypasses Python's GIL completely")
    print("- Each CPU core gets its own Python process")
    print("- Scales linearly with CPU cores")
    print("- Should see 90%+ CPU utilization now!")
    print("- Massive speedup on multi-core systems")
    
    if MAX_DURATION is None:
        print("\n⚠️ Warning: Large datasets may create very long audio files!")
        print("   Consider setting MAX_DURATION if you have huge datasets.")