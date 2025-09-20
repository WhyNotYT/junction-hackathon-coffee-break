import pandas as pd
import json
import os
import numpy as np
from pyod.models.iforest import IForest
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from threading import Lock
import time
import multiprocessing
import warnings

# Suppress sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Thread-safe file writing
file_lock = Lock()

def process_chunk_batch(args):
    """Process multiple chunks in a single worker to maximize CPU usage"""
    chunks_data, scaler_params, training_data, contamination, continuous_cols = args
    
    all_results = []
    
    try:
        # Recreate scaler in worker process
        scaler = StandardScaler()
        scaler.mean_ = np.array(scaler_params['mean'])
        scaler.scale_ = np.array(scaler_params['scale'])
        scaler.var_ = np.array(scaler_params['var'])
        scaler.n_features_in_ = scaler_params['n_features_in_']
        
        # Train a new IsolationForest in each worker with the training data
        clf = IForest(
            n_estimators=100,  # Reduced for speed in each worker
            contamination=contamination,
            random_state=42,
            n_jobs=1,  # Single job per worker to avoid conflicts
            max_samples='auto'
        )
        
        # Fit on the provided training data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(training_data)
        
        # Process each chunk in this batch
        for chunk_data, chunk_indices, chunk_id in chunks_data:
            results = []
            
            # Convert to numpy array and scale (suppress warnings)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scaled_chunk = scaler.transform(chunk_data)
            
            # Get predictions for this chunk
            predictions = clf.predict(scaled_chunk)
            anomaly_mask = predictions == 1
            
            if not np.any(anomaly_mask):
                continue
            
            # Get anomaly indices and decision scores
            anomaly_indices = [chunk_indices[i] for i, is_anomaly in enumerate(anomaly_mask) if is_anomaly]
            decision_scores = clf.decision_function(scaled_chunk)
            
            # Calculate statistics for the chunk
            chunk_df = pd.DataFrame(chunk_data, columns=continuous_cols, index=chunk_indices)
            chunk_mean = chunk_df.mean()
            chunk_std = chunk_df.std()
            
            # Process anomalies
            for i, global_idx in enumerate(anomaly_indices):
                local_idx = chunk_indices.index(global_idx)
                
                # Calculate z-scores
                row_data = chunk_df.loc[global_idx]
                z_scores = (row_data - chunk_mean) / chunk_std
                outlier_columns = z_scores[abs(z_scores) > 3].index.tolist()
                
                if len(outlier_columns) < 1:
                    continue
                
                # Store result
                result = {
                    'global_index': global_idx,
                    'local_index': local_idx,
                    'outlier_columns': outlier_columns,
                    'z_scores': z_scores.to_dict(),
                    'decision_score': float(decision_scores[local_idx]),
                    'chunk_id': chunk_id
                }
                results.append(result)
            
            all_results.extend(results)
            
    except Exception as e:
        print(f"❌ Error processing chunk batch: {e}")
    
    return all_results

def write_flagged_record_batch(records, df, output_file):
    """Thread-safe batch writing of flagged records"""
    with file_lock:
        with open(output_file, 'a', encoding='utf-8') as f:
            for record in records:
                # Get complete row from original dataframe
                idx = record['global_index']
                complete_row = df.iloc[idx]
                
                # Create JSON object
                flagged_record = {
                    "row_index": int(idx),
                    "timestamp": datetime.now().isoformat(),
                    "anomaly_type": "statistical_outlier",
                    "flagged_columns": record['outlier_columns'],
                    "anomaly_scores": {col: float(abs(record['z_scores'][col])) 
                                     for col in record['outlier_columns']},
                    "isolation_forest_score": record['decision_score'],
                    "chunk_id": record['chunk_id'],
                    "data": {}
                }
                
                # Add all column data
                for col, value in complete_row.items():
                    if pd.isna(value):
                        flagged_record["data"][col] = None
                    elif isinstance(value, (int, float)):
                        flagged_record["data"][col] = float(value) if not pd.isna(value) else None
                    else:
                        flagged_record["data"][col] = str(value)
                
                # Write JSON line
                f.write(json.dumps(flagged_record) + '\n')

def detect_anomalies_in_file_multithreaded(file_path, contamination=0.05, output_file="flagged.txt", 
                                         max_workers=None, chunk_size=500, use_processes=True, 
                                         chunks_per_worker=4):
    """
    High-performance anomaly detection with process/thread pooling
    
    Args:
        file_path: Path to CSV file
        contamination: Anomaly contamination rate
        output_file: Output file for flagged records
        max_workers: Number of workers (None = auto-detect)
        chunk_size: Size of data chunks for processing
        use_processes: Use ProcessPoolExecutor (True) vs ThreadPoolExecutor (False)
        chunks_per_worker: Number of chunks to batch per worker
    """
    print(f"🔍 Processing file: {file_path}")
    start_time = time.time()
    
    try:
        # Load data
        print("📂 Loading data...")
        df = pd.read_csv(file_path)
        print(f"📊 Loaded {len(df)} rows, {len(df.columns)} columns")
        
        numeric_df = df.select_dtypes(include=["number"])
        if numeric_df.empty:
            print("⚠️ No numeric columns to analyze.")
            return

        # Exclude binary columns
        continuous_cols = [
            col for col in numeric_df.columns 
            if numeric_df[col].nunique() > 2
        ]
        if not continuous_cols:
            print("⚠️ No continuous numeric columns to analyze.")
            return

        print(f"🔢 Analyzing {len(continuous_cols)} continuous columns: {continuous_cols}")
        
        # Prepare data
        numeric_cont_df = numeric_df[continuous_cols].fillna(numeric_df[continuous_cols].median())
        
        # Train scaler and prepare training data for workers
        print("🔧 Training anomaly detection model...")
        sample_size = min(15000, len(numeric_cont_df))
        sample_indices = np.random.choice(len(numeric_cont_df), sample_size, replace=False)
        sample_data = numeric_cont_df.iloc[sample_indices]
        
        scaler = StandardScaler()
        training_data = scaler.fit_transform(sample_data)
        
        # Serialize scaler parameters for multiprocessing
        scaler_params = {
            'mean': scaler.mean_.tolist(),
            'scale': scaler.scale_.tolist(),
            'var': scaler.var_.tolist(),
            'n_features_in_': scaler.n_features_in_
        }
        
        # Clear output file
        with open(output_file, 'w') as f:
            pass
        
        # Determine optimal number of workers - be more aggressive
        if max_workers is None:
            max_workers = multiprocessing.cpu_count() * 2  # Over-subscribe for I/O bound tasks
        
        print(f"⚡ Using {'processes' if use_processes else 'threads'}: {max_workers} workers")
        print(f"📦 Chunk size: {chunk_size}, Chunks per worker: {chunks_per_worker}")
        
        # Split data into smaller chunks for better parallelization
        total_rows = len(numeric_cont_df)
        chunks = []
        
        for i in range(0, total_rows, chunk_size):
            end_idx = min(i + chunk_size, total_rows)
            chunk_data = numeric_cont_df.iloc[i:end_idx].values
            chunk_indices = numeric_cont_df.iloc[i:end_idx].index.tolist()
            chunk_id = i // chunk_size
            chunks.append((chunk_data, chunk_indices, chunk_id))
        
        # Group chunks into batches for workers
        chunk_batches = []
        for i in range(0, len(chunks), chunks_per_worker):
            batch = chunks[i:i + chunks_per_worker]
            batch_args = (batch, scaler_params, training_data, contamination, continuous_cols)
            chunk_batches.append(batch_args)
        
        print(f"📦 Created {len(chunk_batches)} work batches from {len(chunks)} chunks")
        
        # Choose executor type
        ExecutorClass = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
        context_args = {'max_workers': max_workers}
        
        if use_processes:
            # For processes, we can set additional context
            context_args['mp_context'] = multiprocessing.get_context('spawn')
        
        # Process chunk batches in parallel
        total_anomalies = 0
        processed_batches = 0
        
        with ExecutorClass(**context_args) as executor:
            # Submit all chunk batches
            future_to_batch = {executor.submit(process_chunk_batch, batch): i 
                              for i, batch in enumerate(chunk_batches)}
            
            # Process results as they complete
            for future in as_completed(future_to_batch):
                batch_id = future_to_batch[future]
                try:
                    batch_results = future.result()
                    if batch_results:
                        # Write results in batch
                        write_flagged_record_batch(batch_results, df, output_file)
                        total_anomalies += len(batch_results)
                        print(f"✅ Batch {batch_id}: {len(batch_results)} anomalies detected")
                    
                    processed_batches += 1
                    progress = (processed_batches / len(chunk_batches)) * 100
                    print(f"📊 Progress: {progress:.1f}% ({processed_batches}/{len(chunk_batches)} batches)")
                        
                except Exception as e:
                    print(f"❌ Error processing batch {batch_id}: {e}")
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        if total_anomalies == 0:
            print("✅ No anomalies detected.")
            return
        
        print(f"\n🎉 Processing Complete!")
        print(f"⚠️ Found {total_anomalies} anomalous rows")
        print(f"⏱️ Processing time: {processing_time:.2f} seconds")
        print(f"🚀 Speed: {len(df)/processing_time:.0f} rows/second")
        print(f"📝 Results saved to '{output_file}'")
        print(f"💾 File size: {os.path.getsize(output_file)} bytes")

    except Exception as e:
        print(f"❌ Error processing file: {e}")

def analyze_flagged_file_fast(flagged_file="flagged.txt"):
    """Fast analysis of flagged data file with threading"""
    try:
        if not os.path.exists(flagged_file):
            print(f"❌ File {flagged_file} not found.")
            return
        
        print(f"📊 Analyzing flagged data from: {flagged_file}")
        
        def parse_lines(lines):
            """Parse a batch of lines"""
            results = []
            for line_num, line in lines:
                try:
                    record = json.loads(line.strip())
                    results.append(record)
                except json.JSONDecodeError as e:
                    print(f"⚠️ Invalid JSON on line {line_num}: {e}")
            return results
        
        # Read file in batches
        flagged_data = []
        batch_size = 1000
        
        with open(flagged_file, 'r', encoding='utf-8') as f:
            lines = [(i+1, line) for i, line in enumerate(f)]
        
        # Process in parallel batches
        with ThreadPoolExecutor(max_workers=4) as executor:
            batch_futures = []
            for i in range(0, len(lines), batch_size):
                batch = lines[i:i+batch_size]
                future = executor.submit(parse_lines, batch)
                batch_futures.append(future)
            
            for future in as_completed(batch_futures):
                flagged_data.extend(future.result())
        
        if not flagged_data:
            print("⚠️ No valid flagged records found.")
            return
        
        print(f"✅ Loaded {len(flagged_data)} flagged records")
        
        # Summary statistics
        all_flagged_cols = []
        total_score = 0
        chunk_stats = {}
        
        for record in flagged_data:
            all_flagged_cols.extend(record.get('flagged_columns', []))
            scores = record.get('anomaly_scores', {})
            if scores:
                total_score += max(scores.values())
            
            chunk_id = record.get('chunk_id', 'unknown')
            chunk_stats[chunk_id] = chunk_stats.get(chunk_id, 0) + 1
        
        from collections import Counter
        col_counts = Counter(all_flagged_cols)
        
        print("\n📈 Most frequently flagged columns:")
        for col, count in col_counts.most_common(5):
            print(f"  {col}: {count} times ({count/len(flagged_data)*100:.1f}%)")
        
        print(f"\n🔢 Average anomaly score: {total_score/len(flagged_data):.2f}")
        print(f"📦 Anomalies found across {len(chunk_stats)} chunks")
        
        return flagged_data
    
    except Exception as e:
        print(f"❌ Error analyzing flagged file: {e}")

if __name__ == "__main__":
    file_path = "../Datasets/home-credit-default-risk/credit_card_balance.csv"  # 🔧 change to your CSV
    output_file = "flagged.txt"
    
    # More conservative configuration to avoid model serialization issues
    MAX_WORKERS = multiprocessing.cpu_count() * 2  # Still aggressive but not as extreme
    CHUNK_SIZE = 500  # Slightly larger chunks for better efficiency
    CHUNKS_PER_WORKER = 3  # More chunks per worker to reduce model training overhead
    CONTAMINATION = 0.05
    USE_PROCESSES = True  # Use processes for true parallelism (bypasses GIL)
    
    print(f"🚀 Starting high-performance anomaly detection...")
    print(f"💻 System info: {multiprocessing.cpu_count()} CPU cores detected")
    print(f"⚙️ Configuration: {MAX_WORKERS} workers, {CHUNK_SIZE} chunk size, {CHUNKS_PER_WORKER} chunks/worker")
    print(f"🔄 Using {'multiprocessing' if USE_PROCESSES else 'multithreading'}")
    
    # Detect anomalies with aggressive parallelization
    detect_anomalies_in_file_multithreaded(
        file_path, 
        contamination=CONTAMINATION,
        output_file=output_file,
        max_workers=MAX_WORKERS,
        chunk_size=CHUNK_SIZE,
        use_processes=USE_PROCESSES,
        chunks_per_worker=CHUNKS_PER_WORKER
    )
    
    # Analyze the flagged data
    print("\n" + "="*50)
    analyze_flagged_file_fast(output_file)