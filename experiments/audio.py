import pandas as pd
import json
import numpy as np
import wave
import struct
import os
from collections import defaultdict
import warnings

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
    
    # Remove NaN values
    clean_data = data[~np.isnan(data)]
    if len(clean_data) == 0:
        return np.zeros_like(data)
    
    # Get data range
    data_min, data_max = np.min(clean_data), np.max(clean_data)
    
    # Avoid division by zero
    if data_max == data_min:
        return np.full_like(data, target_range[0])
    
    # Normalize to target range
    normalized = (data - data_min) / (data_max - data_min)
    normalized = normalized * (target_range[1] - target_range[0]) + target_range[0]
    
    # Handle NaN values by setting them to target_range[0]
    normalized[np.isnan(normalized)] = target_range[0]
    
    return normalized

def create_wav_file(data, filename, sample_rate=44100, duration=5.0):
    """
    Create a WAV file from numerical data
    
    Args:
        data: numpy array of normalized data (-1.0 to 1.0)
        filename: output WAV filename
        sample_rate: audio sample rate in Hz
        duration: desired duration in seconds
    """
    if len(data) == 0:
        print(f"⚠️ No data to convert for {filename}")
        return
    
    # Calculate number of samples needed
    num_samples = int(sample_rate * duration)
    
    # Interpolate data to match desired sample count
    if len(data) != num_samples:
        x_original = np.linspace(0, 1, len(data))
        x_new = np.linspace(0, 1, num_samples)
        audio_data = np.interp(x_new, x_original, data)
    else:
        audio_data = data.copy()
    
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
        
        print(f"✅ Created: {filename} ({duration}s, {sample_rate}Hz)")
        
    except Exception as e:
        print(f"❌ Error creating {filename}: {e}")

def add_anomaly_highlights(normal_data, anomaly_indices, anomaly_values, highlight_factor=2.0):
    """
    Add highlights to audio data where anomalies occur
    
    Args:
        normal_data: normalized normal data
        anomaly_indices: list of indices where anomalies occur
        anomaly_values: list of anomaly values
        highlight_factor: multiplier for anomaly emphasis
    
    Returns:
        modified audio data with anomaly highlights
    """
    if len(normal_data) == 0 or len(anomaly_indices) == 0:
        return normal_data
    
    highlighted_data = normal_data.copy()
    
    # Map anomaly indices to audio data indices
    data_length = len(normal_data)
    
    for i, (orig_idx, anom_val) in enumerate(zip(anomaly_indices, anomaly_values)):
        # Map original data index to audio data index
        audio_idx = int((orig_idx / len(anomaly_indices)) * data_length) if len(anomaly_indices) > 1 else data_length // 2
        
        # Create a highlight region around the anomaly
        highlight_width = max(1, data_length // 100)  # 1% of data length
        start_idx = max(0, audio_idx - highlight_width // 2)
        end_idx = min(data_length, audio_idx + highlight_width // 2)
        
        # Amplify the region (but keep within bounds)
        for j in range(start_idx, end_idx):
            highlighted_data[j] = np.clip(highlighted_data[j] * highlight_factor, -1.0, 1.0)
    
    return highlighted_data

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
                        sample_rate=44100, duration=5.0, create_anomaly_highlights=True):
    """
    Export CSV columns and flagged data to WAV audio files
    
    Args:
        csv_file: path to original CSV data
        flagged_file: path to flagged anomaly data (JSON lines)
        output_dir: directory to save WAV files
        sample_rate: audio sample rate
        duration: duration of each audio file in seconds
        create_anomaly_highlights: whether to create highlighted versions
    """
    print(f"🎵 Converting data to audio files...")
    print(f"📁 Input CSV: {csv_file}")
    print(f"📁 Flagged data: {flagged_file}")
    print(f"📁 Output directory: {output_dir}")
    
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
                flagged_by_column[col].append(data[col])
                flagged_indices_by_column[col].append(row_idx)
    
    # Get numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"🔢 Found {len(numeric_columns)} numeric columns: {numeric_columns}")
    
    # Export each numeric column as audio
    for col in numeric_columns:
        print(f"\n🎼 Processing column: {col}")
        
        # Get column data (fill NaN with median)
        col_data = df[col].fillna(df[col].median()).values
        
        if len(col_data) == 0:
            print(f"⚠️ No data in column {col}")
            continue
        
        # Normalize data for audio
        normalized_data = normalize_data_for_audio(col_data)
        
        # Create standard audio file
        standard_filename = os.path.join(output_dir, f"{col}_standard.wav")
        create_wav_file(normalized_data, standard_filename, sample_rate, duration)
        
        # Create anomaly-highlighted version if anomalies exist for this column
        if create_anomaly_highlights and col in flagged_by_column:
            flagged_values = flagged_by_column[col]
            flagged_indices = flagged_indices_by_column[col]
            
            print(f"  🚨 Found {len(flagged_values)} anomalies in {col}")
            
            # Create highlighted version
            highlighted_data = add_anomaly_highlights(
                normalized_data, flagged_indices, flagged_values
            )
            
            highlighted_filename = os.path.join(output_dir, f"{col}_with_anomalies.wav")
            create_wav_file(highlighted_data, highlighted_filename, sample_rate, duration)
            
            # Create anomalies-only audio (just the flagged values)
            if len(flagged_values) > 1:
                flagged_normalized = normalize_data_for_audio(np.array(flagged_values))
                anomalies_only_filename = os.path.join(output_dir, f"{col}_anomalies_only.wav")
                create_wav_file(flagged_normalized, anomalies_only_filename, sample_rate, duration)
    
    print(f"\n🎉 Audio export complete!")
    print(f"📁 Files saved to: {output_dir}")
    print(f"🔊 Audio format: {sample_rate}Hz, {duration}s duration")
    
    # Create summary file
    summary_file = os.path.join(output_dir, "audio_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("Audio Export Summary\n")
        f.write("==================\n\n")
        f.write(f"Original CSV: {csv_file}\n")
        f.write(f"Flagged data: {flagged_file}\n")
        f.write(f"Processed columns: {len(numeric_columns)}\n")
        f.write(f"Total anomalies: {len(flagged_records)}\n")
        f.write(f"Sample rate: {sample_rate}Hz\n")
        f.write(f"Duration: {duration}s\n\n")
        
        f.write("File Types Created:\n")
        f.write("- *_standard.wav: Normal data sonification\n")
        f.write("- *_with_anomalies.wav: Data with anomalies emphasized\n")
        f.write("- *_anomalies_only.wav: Only the anomalous values\n\n")
        
        f.write("Columns with anomalies:\n")
        for col, values in flagged_by_column.items():
            f.write(f"- {col}: {len(values)} anomalies\n")
    
    print(f"📝 Summary saved to: {summary_file}")

# Example usage and configuration
if __name__ == "__main__":
    # Configuration - Update these paths to match your files
    CSV_FILE = "../Datasets/home-credit-default-risk/credit_card_balance.csv"  # Your original CSV
    FLAGGED_FILE = "../experiments/flagged.txt"  # Your flagged anomalies file
    OUTPUT_DIR = "audio_analysis"  # Output directory for WAV files
    
    # Audio settings
    SAMPLE_RATE = 44100  # CD quality
    DURATION = 5.0       # 5 seconds per audio file
    CREATE_HIGHLIGHTS = True  # Create anomaly-highlighted versions
    
    print("🎵 Data-to-Audio Converter for Anomaly Analysis")
    print("=" * 50)
    
    # Run the conversion
    export_data_to_audio(
        csv_file=CSV_FILE,
        flagged_file=FLAGGED_FILE,
        output_dir=OUTPUT_DIR,
        sample_rate=SAMPLE_RATE,
        duration=DURATION,
        create_anomaly_highlights=CREATE_HIGHLIGHTS
    )
    
    print("\n🎧 Listening Guide:")
    print("- Standard files: Hear the normal data patterns")
    print("- With anomalies: Anomalies are amplified/emphasized")
    print("- Anomalies only: Just the outlier values as audio")
    print("- Higher pitches = higher values")
    print("- Sudden volume changes = potential anomalies")
    print("- Listen for unusual patterns, spikes, or rhythm changes")