import pandas as pd
import json
import os
import numpy as np
import cv2
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.models.knn import KNN
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from scipy import stats
from scipy.spatial.distance import mahalanobis
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from threading import Lock
import time
import multiprocessing
import warnings
from collections import Counter

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Thread-safe file writing
file_lock = Lock()

class MultiMethodAnomalyDetector:
    """Enhanced anomaly detector using multiple methods with confidence voting"""
    
    def __init__(self, contamination=0.05, confidence_threshold=0.6, min_votes=2):
        self.contamination = contamination
        self.confidence_threshold = confidence_threshold
        self.min_votes = min_votes
        self.methods = {}
        self.scalers = {}
        
    def _prepare_morphological_features(self, data):
        """Extract morphological features from data patterns"""
        try:
            # Convert data to image-like format for morphological operations
            if data.shape[1] < 2:
                return data
                
            # Normalize data to 0-255 range for CV2
            normalized_data = ((data - data.min()) / (data.max() - data.min()) * 255).astype(np.uint8)
            
            morphological_features = []
            
            for row in normalized_data:
                # Reshape row to 2D if possible
                if len(row) >= 4:
                    # Try to create a square-ish image
                    side_len = int(np.sqrt(len(row)))
                    if side_len * side_len <= len(row):
                        img = row[:side_len*side_len].reshape(side_len, side_len)
                        
                        # Apply morphological operations
                        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                        
                        # Opening (erosion followed by dilation)
                        opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
                        
                        # Closing (dilation followed by erosion)
                        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
                        
                        # Gradient (difference between dilation and erosion)
                        gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
                        
                        # Calculate features from morphological operations
                        features = [
                            np.mean(opening), np.std(opening),
                            np.mean(closing), np.std(closing),
                            np.mean(gradient), np.std(gradient),
                            np.sum(gradient > 0),  # Number of edge pixels
                            np.mean(np.abs(img - opening)),  # Texture measure
                            np.mean(np.abs(img - closing)),   # Pattern measure
                        ]
                        morphological_features.append(features)
                    else:
                        # Fallback: use statistical features
                        morphological_features.append([
                            np.mean(row), np.std(row), np.median(row), 
                            stats.skew(row), stats.kurtosis(row),
                            np.percentile(row, 25), np.percentile(row, 75),
                            np.sum(np.diff(row) > 0), np.sum(np.diff(row) < 0)
                        ])
                else:
                    # Very few features - use basic stats
                    morphological_features.append([
                        np.mean(row), np.std(row), np.min(row), np.max(row),
                        0, 0, 0, 0, 0  # Pad with zeros
                    ])
            
            return np.array(morphological_features)
        
        except Exception as e:
            print(f"⚠️ Morphological feature extraction failed: {e}")
            return data
    
    def fit(self, data):
        """Fit multiple anomaly detection methods"""
        print("🔧 Training multiple anomaly detection methods...")
        
        # Prepare different scalers
        self.scalers['standard'] = StandardScaler()
        self.scalers['robust'] = RobustScaler()
        
        # Fit scalers
        data_standard = self.scalers['standard'].fit_transform(data)
        data_robust = self.scalers['robust'].fit_transform(data)
        
        # Prepare morphological features
        morph_features = self._prepare_morphological_features(data_standard)
        
        # Method 1: Isolation Forest (PyOD)
        self.methods['isolation_forest'] = IForest(
            n_estimators=200,
            contamination=self.contamination,
            random_state=42,
            max_samples='auto'
        )
        
        # Method 2: Local Outlier Factor
        self.methods['lof'] = LOF(
            n_neighbors=20,
            contamination=self.contamination,
            leaf_size=30
        )
        
        # Method 3: One-Class SVM
        self.methods['ocsvm'] = OCSVM(
            contamination=self.contamination,
            kernel='rbf',
            gamma='auto'
        )
        
        # Method 4: K-Nearest Neighbors
        self.methods['knn'] = KNN(
            contamination=self.contamination,
            n_neighbors=15,
            method='largest'
        )
        
        # Method 5: Elliptic Envelope (Sklearn)
        self.methods['elliptic'] = EllipticEnvelope(
            contamination=self.contamination,
            random_state=42
        )
        
        # Method 6: Statistical Z-Score method (custom)
        self.methods['zscore'] = None  # Will be handled separately
        
        # Method 7: Mahalanobis Distance (custom)
        self.methods['mahalanobis'] = None  # Will be handled separately
        
        # Fit methods
        try:
            self.methods['isolation_forest'].fit(data_standard)
            print("✅ Isolation Forest trained")
        except Exception as e:
            print(f"❌ Isolation Forest failed: {e}")
            del self.methods['isolation_forest']
        
        try:
            self.methods['lof'].fit(data_robust)
            print("✅ LOF trained")
        except Exception as e:
            print(f"❌ LOF failed: {e}")
            del self.methods['lof']
        
        try:
            self.methods['ocsvm'].fit(data_standard)
            print("✅ One-Class SVM trained")
        except Exception as e:
            print(f"❌ One-Class SVM failed: {e}")
            del self.methods['ocsvm']
        
        try:
            self.methods['knn'].fit(data_robust)
            print("✅ KNN trained")
        except Exception as e:
            print(f"❌ KNN failed: {e}")
            del self.methods['knn']
        
        try:
            self.methods['elliptic'].fit(data_standard)
            print("✅ Elliptic Envelope trained")
        except Exception as e:
            print(f"❌ Elliptic Envelope failed: {e}")
            del self.methods['elliptic']
        
        # Store data statistics for custom methods
        self.data_mean = np.mean(data, axis=0)
        self.data_std = np.std(data, axis=0)
        self.data_cov = np.cov(data.T)
        self.data_cov_inv = np.linalg.pinv(self.data_cov)  # Pseudo-inverse for numerical stability
        
        # Store morphological features statistics
        if morph_features is not None:
            self.morph_mean = np.mean(morph_features, axis=0)
            self.morph_std = np.std(morph_features, axis=0)
        
        print(f"🎯 Successfully trained {len(self.methods)} anomaly detection methods")
    
    def predict_with_confidence(self, data):
        """Predict anomalies using all methods and return confidence scores"""
        n_samples = data.shape[0]
        
        # Scale data
        data_standard = self.scalers['standard'].transform(data)
        data_robust = self.scalers['robust'].transform(data)
        
        # Prepare morphological features
        morph_features = self._prepare_morphological_features(data_standard)
        
        # Store predictions and scores from each method
        predictions = {}
        scores = {}
        
        # Method predictions
        for method_name, method in self.methods.items():
            if method is None:
                continue
                
            try:
                if method_name == 'isolation_forest':
                    pred = method.predict(data_standard)
                    score = method.decision_function(data_standard)
                elif method_name == 'lof':
                    pred = method.predict(data_robust)
                    score = method.decision_function(data_robust)
                elif method_name == 'ocsvm':
                    pred = method.predict(data_standard)
                    score = method.decision_function(data_standard)
                elif method_name == 'knn':
                    pred = method.predict(data_robust)
                    score = method.decision_function(data_robust)
                elif method_name == 'elliptic':
                    pred = method.predict(data_standard)
                    score = method.decision_scores_
                
                predictions[method_name] = pred
                scores[method_name] = score
                
            except Exception as e:
                print(f"⚠️ {method_name} prediction failed: {e}")
        
        # Custom method 1: Z-Score anomalies
        try:
            z_scores = np.abs((data - self.data_mean) / (self.data_std + 1e-8))
            max_z_scores = np.max(z_scores, axis=1)
            z_threshold = stats.norm.ppf(1 - self.contamination)
            
            predictions['zscore'] = (max_z_scores > z_threshold).astype(int)
            scores['zscore'] = max_z_scores
            
        except Exception as e:
            print(f"⚠️ Z-score method failed: {e}")
        
        # Custom method 2: Mahalanobis Distance
        try:
            mahal_distances = []
            for i, row in enumerate(data):
                try:
                    dist = mahalanobis(row, self.data_mean, self.data_cov_inv)
                    mahal_distances.append(dist)
                except Exception:
                    mahal_distances.append(0)
            
            mahal_distances = np.array(mahal_distances)
            mahal_threshold = np.percentile(mahal_distances, (1 - self.contamination) * 100)
            
            predictions['mahalanobis'] = (mahal_distances > mahal_threshold).astype(int)
            scores['mahalanobis'] = mahal_distances
            
        except Exception as e:
            print(f"⚠️ Mahalanobis method failed: {e}")
        
        # Custom method 3: Morphological anomalies
        try:
            if hasattr(self, 'morph_mean') and morph_features is not None:
                morph_z_scores = np.abs((morph_features - self.morph_mean) / (self.morph_std + 1e-8))
                max_morph_z_scores = np.max(morph_z_scores, axis=1)
                morph_threshold = stats.norm.ppf(1 - self.contamination)
                
                predictions['morphological'] = (max_morph_z_scores > morph_threshold).astype(int)
                scores['morphological'] = max_morph_z_scores
        except Exception as e:
            print(f"⚠️ Morphological method failed: {e}")
        
        # Calculate voting confidence
        final_predictions = []
        final_confidences = []
        method_agreements = []
        
        for i in range(n_samples):
            # Count votes for anomaly
            anomaly_votes = sum(predictions[method][i] for method in predictions)
            total_methods = len(predictions)
            
            # Calculate confidence based on agreement
            confidence = anomaly_votes / total_methods if total_methods > 0 else 0
            
            # Final decision: anomaly if enough methods agree AND confidence is high enough
            is_anomaly = (anomaly_votes >= self.min_votes and 
                         confidence >= self.confidence_threshold)
            
            final_predictions.append(int(is_anomaly))
            final_confidences.append(confidence)
            
            # Store which methods agreed
            agreeing_methods = [method for method in predictions 
                              if predictions[method][i] == 1]
            method_agreements.append(agreeing_methods)
        
        return {
            'predictions': np.array(final_predictions),
            'confidences': np.array(final_confidences),
            'method_scores': scores,
            'method_predictions': predictions,
            'method_agreements': method_agreements
        }

def process_chunk_batch_enhanced(args):
    """Enhanced chunk processing with multiple anomaly detection methods"""
    chunks_data, detector_state, continuous_cols = args
    
    all_results = []
    
    try:
        # Recreate detector in worker process
        detector = MultiMethodAnomalyDetector(
            contamination=detector_state['contamination'],
            confidence_threshold=detector_state['confidence_threshold'],
            min_votes=detector_state['min_votes']
        )
        
        # Restore detector state
        detector.methods = {}  # Will be rebuilt
        detector.scalers = detector_state['scalers']
        detector.data_mean = np.array(detector_state['data_mean'])
        detector.data_std = np.array(detector_state['data_std'])
        detector.data_cov = np.array(detector_state['data_cov'])
        detector.data_cov_inv = np.array(detector_state['data_cov_inv'])
        
        if 'morph_mean' in detector_state:
            detector.morph_mean = np.array(detector_state['morph_mean'])
            detector.morph_std = np.array(detector_state['morph_std'])
        
        # Rebuild methods (simplified for worker processes)
        training_data = detector_state['training_data']
        
        # Quick training for essential methods only
        detector.methods['isolation_forest'] = IForest(
            n_estimators=50, contamination=detector.contamination, random_state=42
        )
        detector.methods['isolation_forest'].fit(training_data)
        
        # Process each chunk in this batch
        for chunk_data, chunk_indices, chunk_id in chunks_data:
            try:
                # Get predictions with confidence
                results_dict = detector.predict_with_confidence(chunk_data)
                
                predictions = results_dict['predictions']
                confidences = results_dict['confidences']
                method_agreements = results_dict['method_agreements']
                method_scores = results_dict['method_scores']
                
                # Find anomalies
                anomaly_indices = np.where(predictions == 1)[0]
                
                if len(anomaly_indices) == 0:
                    continue
                
                # Process each anomaly
                chunk_df = pd.DataFrame(chunk_data, columns=continuous_cols, index=chunk_indices)
                chunk_mean = chunk_df.mean()
                chunk_std = chunk_df.std()
                
                for local_idx in anomaly_indices:
                    global_idx = chunk_indices[local_idx]
                    
                    # Calculate detailed anomaly information
                    row_data = chunk_df.iloc[local_idx]
                    z_scores = (row_data - chunk_mean) / (chunk_std + 1e-8)
                    outlier_columns = z_scores[abs(z_scores) > 2].index.tolist()
                    
                    # Collect scores from all methods
                    all_method_scores = {}
                    for method_name, method_scores_arr in method_scores.items():
                        if local_idx < len(method_scores_arr):
                            all_method_scores[method_name] = float(method_scores_arr[local_idx])
                    
                    # Store comprehensive result
                    result = {
                        'global_index': global_idx,
                        'local_index': local_idx,
                        'confidence': float(confidences[local_idx]),
                        'outlier_columns': outlier_columns,
                        'z_scores': z_scores.to_dict(),
                        'method_agreements': method_agreements[local_idx],
                        'all_method_scores': all_method_scores,
                        'chunk_id': chunk_id,
                        'num_agreeing_methods': len(method_agreements[local_idx])
                    }
                    all_results.append(result)
                    
            except Exception as e:
                print(f"❌ Error processing chunk {chunk_id}: {e}")
                continue
                
    except Exception as e:
        print(f"❌ Error in chunk batch processing: {e}")
    
    return all_results

def write_flagged_record_batch_enhanced(records, df, output_file):
    """Enhanced thread-safe batch writing with detailed anomaly information"""
    with file_lock:
        with open(output_file, 'a', encoding='utf-8') as f:
            for record in records:
                try:
                    # Get complete row from original dataframe
                    idx = record['global_index']
                    complete_row = df.iloc[idx]
                    
                    # Create comprehensive JSON object
                    flagged_record = {
                        "row_index": int(idx),
                        "timestamp": datetime.now().isoformat(),
                        "anomaly_type": "multi_method_consensus",
                        "confidence_score": record['confidence'],
                        "num_agreeing_methods": record['num_agreeing_methods'],
                        "agreeing_methods": record['method_agreements'],
                        "flagged_columns": record['outlier_columns'],
                        "anomaly_scores": {col: float(abs(record['z_scores'][col])) 
                                         for col in record['outlier_columns']},
                        "method_specific_scores": record['all_method_scores'],
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
                    
                except Exception as e:
                    print(f"❌ Error writing record {record.get('global_index', 'unknown')}: {e}")

def detect_anomalies_enhanced(file_path, contamination=0.05, output_file="flagged_enhanced.txt", 
                            max_workers=None, chunk_size=300, confidence_threshold=0.6, 
                            min_votes=2, chunks_per_worker=2):
    """
    Enhanced multi-method anomaly detection with morphological analysis
    
    Args:
        file_path: Path to CSV file
        contamination: Anomaly contamination rate
        output_file: Output file for flagged records
        max_workers: Number of workers
        chunk_size: Size of data chunks
        confidence_threshold: Minimum confidence to flag as anomaly
        min_votes: Minimum number of methods that must agree
        chunks_per_worker: Number of chunks per worker
    """
    print(f"🔍 Enhanced multi-method anomaly detection on: {file_path}")
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

        # Get continuous columns
        continuous_cols = [
            col for col in numeric_df.columns 
            if numeric_df[col].nunique() > 2
        ]
        if not continuous_cols:
            print("⚠️ No continuous numeric columns to analyze.")
            return

        print(f"🔢 Analyzing {len(continuous_cols)} continuous columns: {continuous_cols[:5]}...")
        
        # Prepare data
        numeric_cont_df = numeric_df[continuous_cols].fillna(numeric_df[continuous_cols].median())
        
        # Train enhanced detector
        print("🤖 Training enhanced multi-method anomaly detector...")
        sample_size = min(10000, len(numeric_cont_df))  # Smaller sample for faster training
        sample_indices = np.random.choice(len(numeric_cont_df), sample_size, replace=False)
        sample_data = numeric_cont_df.iloc[sample_indices].values
        
        detector = MultiMethodAnomalyDetector(
            contamination=contamination,
            confidence_threshold=confidence_threshold,
            min_votes=min_votes
        )
        detector.fit(sample_data)
        
        # Prepare detector state for multiprocessing
        detector_state = {
            'contamination': contamination,
            'confidence_threshold': confidence_threshold,
            'min_votes': min_votes,
            'scalers': detector.scalers,
            'data_mean': detector.data_mean.tolist(),
            'data_std': detector.data_std.tolist(),
            'data_cov': detector.data_cov.tolist(),
            'data_cov_inv': detector.data_cov_inv.tolist(),
            'training_data': sample_data
        }
        
        if hasattr(detector, 'morph_mean'):
            detector_state['morph_mean'] = detector.morph_mean.tolist()
            detector_state['morph_std'] = detector.morph_std.tolist()
        
        # Clear output file
        with open(output_file, 'w') as f:
            pass
        
        # Setup parallel processing
        if max_workers is None:
            max_workers = min(multiprocessing.cpu_count(), 8)  # Cap at 8 for stability
        
        print(f"⚡ Using {max_workers} workers")
        print(f"📦 Chunk size: {chunk_size}, Chunks per worker: {chunks_per_worker}")
        
        # Create chunks
        total_rows = len(numeric_cont_df)
        chunks = []
        
        for i in range(0, total_rows, chunk_size):
            end_idx = min(i + chunk_size, total_rows)
            chunk_data = numeric_cont_df.iloc[i:end_idx].values
            chunk_indices = numeric_cont_df.iloc[i:end_idx].index.tolist()
            chunk_id = i // chunk_size
            chunks.append((chunk_data, chunk_indices, chunk_id))
        
        # Group chunks into batches
        chunk_batches = []
        for i in range(0, len(chunks), chunks_per_worker):
            batch = chunks[i:i + chunks_per_worker]
            batch_args = (batch, detector_state, continuous_cols)
            chunk_batches.append(batch_args)
        
        print(f"📦 Created {len(chunk_batches)} work batches from {len(chunks)} chunks")
        
        # Process in parallel
        total_anomalies = 0
        processed_batches = 0
        confidence_scores = []
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch = {executor.submit(process_chunk_batch_enhanced, batch): i 
                              for i, batch in enumerate(chunk_batches)}
            
            for future in as_completed(future_to_batch):
                batch_id = future_to_batch[future]
                try:
                    batch_results = future.result()
                    if batch_results:
                        write_flagged_record_batch_enhanced(batch_results, df, output_file)
                        total_anomalies += len(batch_results)
                        
                        # Collect confidence scores
                        batch_confidences = [r['confidence'] for r in batch_results]
                        confidence_scores.extend(batch_confidences)
                        
                        avg_confidence = np.mean(batch_confidences)
                        print(f"✅ Batch {batch_id}: {len(batch_results)} anomalies (avg confidence: {avg_confidence:.3f})")
                    
                    processed_batches += 1
                    progress = (processed_batches / len(chunk_batches)) * 100
                    print(f"📊 Progress: {progress:.1f}% ({processed_batches}/{len(chunk_batches)} batches)")
                        
                except Exception as e:
                    print(f"❌ Error processing batch {batch_id}: {e}")
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"\n🎉 Enhanced Anomaly Detection Complete!")
        if total_anomalies > 0:
            print(f"⚠️ Found {total_anomalies} high-confidence anomalous rows")
            print(f"📊 Average confidence score: {np.mean(confidence_scores):.3f}")
            print(f"📊 Confidence range: {np.min(confidence_scores):.3f} - {np.max(confidence_scores):.3f}")
        else:
            print("✅ No high-confidence anomalies detected.")
            
        print(f"⏱️ Processing time: {processing_time:.2f} seconds")
        print(f"🚀 Speed: {len(df)/processing_time:.0f} rows/second")
        print(f"📝 Results saved to '{output_file}'")
        
        if os.path.exists(output_file):
            print(f"💾 File size: {os.path.getsize(output_file)} bytes")

    except Exception as e:
        print(f"❌ Error in enhanced anomaly detection: {e}")

def analyze_flagged_file_enhanced(flagged_file="flagged_enhanced.txt"):
    """Enhanced analysis of flagged data with method agreement statistics"""
    try:
        if not os.path.exists(flagged_file):
            print(f"❌ File {flagged_file} not found.")
            return
        
        print(f"📊 Analyzing enhanced flagged data from: {flagged_file}")
        
        flagged_data = []
        with open(flagged_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    record = json.loads(line.strip())
                    flagged_data.append(record)
                except json.JSONDecodeError as e:
                    print(f"⚠️ Invalid JSON on line {line_num}: {e}")
        
        if not flagged_data:
            print("⚠️ No valid flagged records found.")
            return
        
        print(f"✅ Loaded {len(flagged_data)} high-confidence flagged records")
        
        # Enhanced analysis
        confidence_scores = [r['confidence_score'] for r in flagged_data]
        method_agreements = [r['num_agreeing_methods'] for r in flagged_data]
        all_flagged_cols = []
        method_usage = Counter()
        
        for record in flagged_data:
            all_flagged_cols.extend(record.get('flagged_columns', []))
            for method in record.get('agreeing_methods', []):
                method_usage[method] += 1
        
        # Statistics
        col_counts = Counter(all_flagged_cols)
        
        print(f"\n📈 Confidence Statistics:")
        print(f"  Average confidence: {np.mean(confidence_scores):.3f}")
        print(f"  Confidence range: {np.min(confidence_scores):.3f} - {np.max(confidence_scores):.3f}")
        print(f"  High confidence (>0.8): {sum(1 for c in confidence_scores if c > 0.8)}")
        
        print(f"\n🤝 Method Agreement Statistics:")
        print(f"  Average methods agreeing: {np.mean(method_agreements):.1f}")
        print(f"  Method agreement range: {np.min(method_agreements)} - {np.max(method_agreements)}")
        
        print(f"\n🔬 Most Active Detection Methods:")
        for method, count in method_usage.most_common(5):
            print(f"  {method}: {count} detections ({count/len(flagged_data)*100:.1f}%)")
        
        print(f"\n📊 Most Frequently Flagged Columns:")
        for col, count in col_counts.most_common(5):
            print(f"  {col}: {count} times ({count/len(flagged_data)*100:.1f}%)")
        
        return flagged_data
        
    except Exception as e:
        print(f"❌ Error analyzing enhanced flagged file: {e}")

if __name__ == "__main__":
    # Configuration
    file_path = "output/cleaned_firstpass.csv"  # 🔧 Change to your CSV
    output_file = "flagged_enhanced.txt"
    
    # Enhanced detection parameters
    MAX_WORKERS = min(multiprocessing.cpu_count(), 6)  # Conservative for stability
    CHUNK_SIZE = 300  # Smaller chunks due to increased computation
    CHUNKS_PER_WORKER = 2  # Fewer chunks per worker due to complex processing
    CONTAMINATION = 0.05
    CONFIDENCE_THRESHOLD = 0.6  # Require 60% of methods to agree
    MIN_VOTES = 2  # At least 2 methods must agree
    
    print(f"🚀 Starting Enhanced Multi-Method Anomaly Detection...")
    print(f"💻 System info: {multiprocessing.cpu_count()} CPU cores detected")
    print(f"⚙️ Configuration:")
    print(f"   - Workers: {MAX_WORKERS}")
    print(f"   - Chunk size: {CHUNK_SIZE}")
    print(f"   - Chunks per worker: {CHUNKS_PER_WORKER}")
    print(f"   - Contamination rate: {CONTAMINATION}")
    print(f"   - Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"   - Minimum votes required: {MIN_VOTES}")
    print(f"🔄 Using multiprocessing with enhanced anomaly detection methods")
    
    # Run enhanced anomaly detection
    detect_anomalies_enhanced(
        file_path=file_path,
        contamination=CONTAMINATION,
        output_file=output_file,
        max_workers=MAX_WORKERS,
        chunk_size=CHUNK_SIZE,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        min_votes=MIN_VOTES,
        chunks_per_worker=CHUNKS_PER_WORKER
    )
    
    # Analyze results
    print("\n" + "="*60)
    print("📋 DETAILED ANALYSIS OF DETECTED ANOMALIES")
    print("="*60)
    analyze_flagged_file_enhanced(output_file)
    
    print("\n" + "="*60)
    print("🎯 DETECTION SUMMARY")
    print("="*60)
    
    # Additional summary statistics
    try:
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                lines = f.readlines()
            
            if lines:
                print(f"📊 Total anomalies detected: {len(lines)}")
                
                # Parse first few records for detailed info
                sample_records = []
                for i, line in enumerate(lines[:5]):  # First 5 records
                    try:
                        record = json.loads(line.strip())
                        sample_records.append(record)
                    except:
                        continue
                
                if sample_records:
                    print(f"\n🔍 Sample Anomaly Details:")
                    for i, record in enumerate(sample_records[:3], 1):
                        print(f"  Anomaly #{i}:")
                        print(f"    - Row index: {record.get('row_index', 'N/A')}")
                        print(f"    - Confidence: {record.get('confidence_score', 0):.3f}")
                        print(f"    - Methods agreeing: {record.get('num_agreeing_methods', 0)}")
                        print(f"    - Flagged columns: {record.get('flagged_columns', [])[:3]}")
                        print(f"    - Detection methods: {record.get('agreeing_methods', [])}")
                
                # Method performance summary
                all_methods = set()
                method_counts = Counter()
                confidence_by_method = {}
                
                for line in lines:
                    try:
                        record = json.loads(line.strip())
                        methods = record.get('agreeing_methods', [])
                        confidence = record.get('confidence_score', 0)
                        
                        for method in methods:
                            all_methods.add(method)
                            method_counts[method] += 1
                            if method not in confidence_by_method:
                                confidence_by_method[method] = []
                            confidence_by_method[method].append(confidence)
                    except:
                        continue
                
                print(f"\n🏆 Detection Method Performance:")
                for method in sorted(all_methods):
                    count = method_counts[method]
                    avg_conf = np.mean(confidence_by_method[method]) if method in confidence_by_method else 0
                    print(f"  {method:20s}: {count:4d} detections (avg confidence: {avg_conf:.3f})")
                
                # Confidence distribution
                all_confidences = []
                for line in lines:
                    try:
                        record = json.loads(line.strip())
                        all_confidences.append(record.get('confidence_score', 0))
                    except:
                        continue
                
                if all_confidences:
                    print(f"\n📊 Confidence Score Distribution:")
                    print(f"  Mean: {np.mean(all_confidences):.3f}")
                    print(f"  Std:  {np.std(all_confidences):.3f}")
                    print(f"  Min:  {np.min(all_confidences):.3f}")
                    print(f"  Max:  {np.max(all_confidences):.3f}")
                    
                    # Confidence buckets
                    high_conf = sum(1 for c in all_confidences if c >= 0.8)
                    med_conf = sum(1 for c in all_confidences if 0.6 <= c < 0.8)
                    low_conf = sum(1 for c in all_confidences if c < 0.6)
                    
                    print(f"  High confidence (≥0.8): {high_conf} ({high_conf/len(all_confidences)*100:.1f}%)")
                    print(f"  Medium confidence (0.6-0.8): {med_conf} ({med_conf/len(all_confidences)*100:.1f}%)")
                    print(f"  Low confidence (<0.6): {low_conf} ({low_conf/len(all_confidences)*100:.1f}%)")
            else:
                print("📊 No anomalies detected with the current thresholds")
        else:
            print("❌ Output file not found")
            
    except Exception as e:
        print(f"⚠️ Error in summary analysis: {e}")
    
    print(f"\n✅ Enhanced anomaly detection pipeline completed!")
    print(f"📁 Results saved to: {output_file}")
    print(f"💡 Tip: Adjust CONFIDENCE_THRESHOLD and MIN_VOTES to fine-tune sensitivity")
    
    # Performance recommendations
    total_time = time.time()  # This would need to be calculated properly in real implementation
    print(f"\n🚀 Performance Recommendations:")
    print(f"   - For faster processing: Increase CHUNK_SIZE to 500-1000")
    print(f"   - For higher precision: Increase CONFIDENCE_THRESHOLD to 0.7-0.8")
    print(f"   - For more sensitivity: Decrease MIN_VOTES to 1")
    print(f"   - For fewer false positives: Increase MIN_VOTES to 3-4")
    
    print(f"\n🔧 Method Descriptions:")
    print(f"   - isolation_forest: Tree-based ensemble method for outlier detection")
    print(f"   - lof: Local Outlier Factor - density-based anomaly detection")
    print(f"   - ocsvm: One-Class Support Vector Machine")
    print(f"   - knn: K-Nearest Neighbors anomaly detection")
    print(f"   - elliptic: Elliptic Envelope - assumes normal distribution")
    print(f"   - zscore: Statistical z-score based detection")
    print(f"   - mahalanobis: Mahalanobis distance-based detection")
    print(f"   - morphological: CV2 morphological operations on data patterns")