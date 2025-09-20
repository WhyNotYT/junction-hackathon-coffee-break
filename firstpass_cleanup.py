import pandas as pd
import json
import numpy as np
from typing import Dict, List, Tuple, Any
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count
import time


def process_chunk_numeric_check(args: Tuple[pd.DataFrame, Dict, int, int]) -> List[Tuple[int, bool, List[str]]]:
    """
    Process a chunk of data for numeric checking - only check columns that have numeric rules.
    
    Args:
        args: Tuple of (data_chunk, rules, start_idx, end_idx)
    
    Returns:
        List of results for each row in chunk
    """
    data_chunk, rules, start_idx, end_idx = args
    results = []
    
    # Identify which columns should be numeric based on rules
    numeric_rule_columns = []
    for column, rule in rules.items():
        if isinstance(rule, list) and len(rule) >= 3:
            lower_limit, upper_limit, confidence = rule[0], rule[1], rule[2]
            # If either limit is a number, this column should be numeric
            if (lower_limit is not None and isinstance(lower_limit, (int, float))) or \
               (upper_limit is not None and isinstance(upper_limit, (int, float))):
                numeric_rule_columns.append(column)
    
    for i, (_, row) in enumerate(data_chunk.iterrows()):
        actual_idx = start_idx + i
        non_numeric_columns = []
        
        # Only check columns that are supposed to be numeric
        for column in numeric_rule_columns:
            if column in row.index:
                value = row[column]
                if pd.isna(value):
                    continue  # NaN is considered numeric
                try:
                    float(value)
                except (ValueError, TypeError):
                    non_numeric_columns.append(f"{column}='{value}'")
        
        has_non_numeric = len(non_numeric_columns) > 0
        results.append((actual_idx, has_non_numeric, non_numeric_columns))
    
    return results


def process_chunk_rules(args: Tuple[pd.DataFrame, Dict, int]) -> List[Tuple[int, str, List[str]]]:
    """
    Process a chunk of data for rules checking.
    
    Args:
        args: Tuple of (data_chunk, rules, start_idx)
    
    Returns:
        List of results for each row in chunk
    """
    data_chunk, rules, start_idx = args
    results = []
    
    for i, (_, row) in enumerate(data_chunk.iterrows()):
        actual_idx = start_idx + i
        violations = []
        max_confidence = 0.0
        
        for column, rule in rules.items():
            if column not in row.index:
                continue
                
            if not isinstance(rule, list) or len(rule) < 3:
                continue
                
            lower_limit, upper_limit, confidence = rule[0], rule[1], rule[2]
            value = row[column]
            
            # Skip NaN values for rule checking
            if pd.isna(value):
                continue
            
            # Check if this is a numeric rule (has numeric limits)
            is_numeric_rule = (lower_limit is not None and isinstance(lower_limit, (int, float))) or \
                             (upper_limit is not None and isinstance(upper_limit, (int, float)))
            
            if is_numeric_rule:
                # For numeric rules, try to convert value to numeric
                try:
                    numeric_value = float(value)
                except (ValueError, TypeError):
                    # This shouldn't happen as non-numeric values should be filtered out earlier
                    continue
                
                # Check if numeric value violates limits
                violates = False
                reason = []
                
                if lower_limit is not None and numeric_value < lower_limit:
                    violates = True
                    reason.append(f"{column} < {lower_limit}")
                
                if upper_limit is not None and numeric_value > upper_limit:
                    violates = True
                    reason.append(f"{column} > {upper_limit}")
                
                if violates:
                    violations.extend(reason)
                    max_confidence = max(max_confidence, confidence)
            else:
                # For text rules, check if value matches expected text values
                # This part can be extended based on how text rules are defined
                # For now, we assume text columns don't have violations
                pass
        
        # Determine decision based on violations and confidence
        if not violations:
            decision = 'cleaned'
        elif max_confidence > 0.7:
            decision = 'fully_discarded'
        else:
            decision = 'unsure'
        
        results.append((actual_idx, decision, violations))
    
    return results


class MultiThreadedDatabaseCleaner:
    """
    A multithreaded database cleaner that processes CSV files based on JSON filtering rules.
    Uses multiprocessing with optimized chunking to maximize CPU usage.
    Handles both numeric and text columns appropriately.
    """
    
    def __init__(self, csv_file_path: str, json_rules_path: str, output_dir: str = ".", 
                 n_processes: int = None, chunk_size: int = None):
        """
        Initialize the database cleaner.
        
        Args:
            csv_file_path: Path to the input CSV file
            json_rules_path: Path to the JSON rules file
            output_dir: Directory to save output files (default: current directory)
            n_processes: Number of processes to use (default: CPU count)
            chunk_size: Size of chunks for processing (auto-calculated if None)
        """
        self.csv_file_path = csv_file_path
        self.json_rules_path = json_rules_path
        self.output_dir = Path(output_dir)
        self.n_processes = n_processes or cpu_count()
        self.chunk_size = chunk_size
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.data = None
        self.rules = None
        self.results = {
            'cleaned': [],
            'fully_discarded': [],
            'unsure': []
        }
    
    def load_data(self) -> None:
        """Load CSV data and JSON rules."""
        try:
            # Load CSV
            print("Loading CSV data...")
            self.data = pd.read_csv(self.csv_file_path)
            print(f"Loaded CSV with {len(self.data)} rows and {len(self.data.columns)} columns")
            
            # Auto-calculate optimal chunk size if not provided
            if self.chunk_size is None:
                # Aim for at least 100 rows per chunk, but ensure we have enough chunks for all processes
                min_chunk_size = 100
                ideal_chunks = self.n_processes * 4  # 4 chunks per process for good load balancing
                self.chunk_size = max(min_chunk_size, len(self.data) // ideal_chunks)
                print(f"Auto-calculated chunk size: {self.chunk_size}")
            
            # Load JSON rules
            with open(self.json_rules_path, 'r') as f:
                self.rules = json.load(f)
            print(f"Loaded rules for {len(self.rules)} columns")
            
            # Analyze rules to understand which columns are numeric vs text
            numeric_columns = []
            text_columns = []
            for column, rule in self.rules.items():
                if isinstance(rule, list) and len(rule) >= 3:
                    lower_limit, upper_limit = rule[0], rule[1]
                    if (lower_limit is not None and isinstance(lower_limit, (int, float))) or \
                       (upper_limit is not None and isinstance(upper_limit, (int, float))):
                        numeric_columns.append(column)
                    else:
                        text_columns.append(column)
            
            print(f"Identified {len(numeric_columns)} numeric rule columns and {len(text_columns)} text rule columns")
            if numeric_columns:
                print(f"Numeric columns: {', '.join(numeric_columns[:5])}" + ("..." if len(numeric_columns) > 5 else ""))
            if text_columns:
                print(f"Text columns: {', '.join(text_columns[:5])}" + ("..." if len(text_columns) > 5 else ""))
            
        except FileNotFoundError as e:
            raise FileNotFoundError(f"File not found: {e}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")
        except Exception as e:
            raise Exception(f"Error loading data: {e}")
    
    def _create_chunks(self, data: pd.DataFrame) -> List[Tuple[pd.DataFrame, int]]:
        """Create data chunks with their starting indices."""
        chunks = []
        total_rows = len(data)
        
        for start in range(0, total_rows, self.chunk_size):
            end = min(start + self.chunk_size, total_rows)
            chunk_data = data.iloc[start:end].copy()
            chunks.append((chunk_data, start))
        
        return chunks
    
    def _process_numeric_check_parallel(self) -> Tuple[List[int], int]:
        """Check for non-numeric values in numeric rule columns using multiprocessing."""
        print(f"Checking for non-numeric values in numeric columns using {self.n_processes} processes...")
        start_time = time.time()
        
        chunks = self._create_chunks(self.data)
        
        # Prepare arguments for multiprocessing - pass the entire rules dict
        chunk_args = []
        for chunk_data, start_idx in chunks:
            end_idx = start_idx + len(chunk_data)
            chunk_args.append((chunk_data, self.rules, start_idx, end_idx))
        
        print(f"Created {len(chunk_args)} chunks of ~{self.chunk_size} rows each")
        
        non_numeric_indices = []
        
        # Process chunks in parallel
        with Pool(processes=self.n_processes) as pool:
            chunk_results = pool.map(process_chunk_numeric_check, chunk_args)
        
        # Flatten results
        non_numeric_count = 0
        for chunk_result in chunk_results:
            for idx, has_non_numeric, non_numeric_cols in chunk_result:
                if has_non_numeric:
                    non_numeric_indices.append(idx)
                    non_numeric_count += 1
                    if non_numeric_count <= 100:  # Limit output for large datasets
                        print(f"Row {idx} FULLY DISCARDED: Non-numeric values in {', '.join(non_numeric_cols)}")
                    elif non_numeric_count == 101:
                        print("... (limiting output, more non-numeric rows discarded)")
        
        elapsed = time.time() - start_time
        print(f"Non-numeric check completed in {elapsed:.2f}s")
        
        return non_numeric_indices, non_numeric_count
    
    def _prepare_data_for_processing(self, exclude_indices: List[int]) -> pd.DataFrame:
        """Prepare data by converting numeric rule columns to numeric where possible."""
        # Remove non-numeric rows first
        clean_data = self.data.drop(index=exclude_indices).copy()
        
        # Convert only numeric rule columns to numeric
        for column, rule in self.rules.items():
            if column in clean_data.columns:
                if isinstance(rule, list) and len(rule) >= 3:
                    lower_limit, upper_limit = rule[0], rule[1]
                    # Only convert if this is a numeric rule
                    if (lower_limit is not None and isinstance(lower_limit, (int, float))) or \
                       (upper_limit is not None and isinstance(upper_limit, (int, float))):
                        clean_data[column] = pd.to_numeric(clean_data[column], errors='coerce')
        
        return clean_data
    
    def _process_rules_parallel(self, clean_data: pd.DataFrame) -> Dict[str, List[int]]:
        """Process rules checking using multiprocessing."""
        print(f"Processing rules in parallel using {self.n_processes} processes...")
        start_time = time.time()
        
        chunks = self._create_chunks(clean_data)
        
        # Prepare arguments for multiprocessing
        chunk_args = []
        for chunk_data, start_idx in chunks:
            chunk_args.append((chunk_data, self.rules, start_idx))
        
        print(f"Created {len(chunk_args)} chunks of ~{self.chunk_size} rows each")
        
        results_dict = {
            'cleaned': [],
            'fully_discarded': [],
            'unsure': []
        }
        
        # Process chunks in parallel
        with Pool(processes=self.n_processes) as pool:
            chunk_results = pool.map(process_chunk_rules, chunk_args)
        
        # Flatten and process results
        discard_count = 0
        unsure_count = 0
        for chunk_result in chunk_results:
            for idx, decision, reasons in chunk_result:
                results_dict[decision].append(idx)
                
                # Print discard/unsure reasons (limit output for large datasets)
                if decision == 'fully_discarded':
                    discard_count += 1
                    if discard_count <= 100:
                        print(f"Row {idx} FULLY DISCARDED: {', '.join(reasons)}")
                    elif discard_count == 101:
                        print("... (limiting output, more rows fully discarded)")
                elif decision == 'unsure':
                    unsure_count += 1
                    if unsure_count <= 100:
                        print(f"Row {idx} MARKED UNSURE: {', '.join(reasons)}")
                    elif unsure_count == 101:
                        print("... (limiting output, more rows marked unsure)")
        
        elapsed = time.time() - start_time
        print(f"Rules processing completed in {elapsed:.2f}s")
        
        return results_dict
    
    def process_data(self) -> Dict[str, int]:
        """
        Process the data according to the rules using multiprocessing.
        
        Returns:
            Dictionary with counts of processed rows
        """
        if self.data is None or self.rules is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print(f"Starting parallel processing with {self.n_processes} processes...")
        print(f"Chunk size: {self.chunk_size} rows per chunk")
        total_start_time = time.time()
        
        # Step 1: Check for non-numeric values in numeric columns only
        non_numeric_indices, non_numeric_count = self._process_numeric_check_parallel()
        
        # Step 2: Prepare data for rule checking (excluding non-numeric rows)
        print("Preparing data for rule processing...")
        clean_data = self._prepare_data_for_processing(non_numeric_indices)
        print(f"Data prepared: {len(clean_data)} rows remaining after removing non-numeric entries")
        
        # Step 3: Process rules in parallel
        rule_results = self._process_rules_parallel(clean_data)
        
        # Step 4: Combine results
        self.results = {
            'cleaned': rule_results['cleaned'],
            'fully_discarded': rule_results['fully_discarded'] + non_numeric_indices,
            'unsure': rule_results['unsure']
        }
        
        # Generate summary
        summary = {
            'total_rows': len(self.data),
            'cleaned': len(self.results['cleaned']),
            'fully_discarded': len(self.results['fully_discarded']),
            'unsure': len(self.results['unsure']),
            'non_numeric_discards': non_numeric_count
        }
        
        total_elapsed = time.time() - total_start_time
        rows_per_second = len(self.data) / total_elapsed
        print(f"\nTotal processing completed in {total_elapsed:.2f}s")
        print(f"Processed {len(self.data)} rows using {self.n_processes} processes")
        print(f"Performance: {rows_per_second:.0f} rows/second")
        
        return summary
    
    def save_results(self) -> Dict[str, str]:
        """
        Save the processed data to separate CSV files.
        
        Returns:
            Dictionary with paths to saved files
        """
        if not any(self.results.values()):
            raise ValueError("No processed data to save. Call process_data() first.")
        
        print("Saving results...")
        saved_files = {}
        
        # Save cleaned data
        if self.results['cleaned']:
            cleaned_data = self.data.iloc[self.results['cleaned']]
            cleaned_path = self.output_dir / "cleaned_firstpass.csv"
            cleaned_data.to_csv(cleaned_path, index=False)
            saved_files['cleaned'] = str(cleaned_path)
            print(f"Saved {len(cleaned_data)} cleaned rows to {cleaned_path}")
        
        # Save fully discarded data
        if self.results['fully_discarded']:
            discarded_data = self.data.iloc[self.results['fully_discarded']]
            discarded_path = self.output_dir / "fully_discarded.csv"
            discarded_data.to_csv(discarded_path, index=False)
            saved_files['fully_discarded'] = str(discarded_path)
            print(f"Saved {len(discarded_data)} discarded rows to {discarded_path}")
        
        # Save unsure data
        if self.results['unsure']:
            unsure_data = self.data.iloc[self.results['unsure']]
            unsure_path = self.output_dir / "unsure.csv"
            unsure_data.to_csv(unsure_path, index=False)
            saved_files['unsure'] = str(unsure_path)
            print(f"Saved {len(unsure_data)} unsure rows to {unsure_path}")
        
        return saved_files
    
    def clean_database(self) -> Dict[str, Any]:
        """
        Complete pipeline: load data, process, and save results.
        
        Returns:
            Dictionary with processing summary and file paths
        """
        self.load_data()
        summary = self.process_data()
        file_paths = self.save_results()
        
        return {
            'summary': summary,
            'files': file_paths
        }


# Convenience functions for backward compatibility
def clean_database_files(csv_path: str, json_rules_path: str, output_dir: str = ".", 
                        n_processes: int = None, chunk_size: int = None) -> Dict[str, Any]:
    """
    Convenience function to clean database files using multiprocessing.
    
    Args:
        csv_path: Path to input CSV file
        json_rules_path: Path to JSON rules file
        output_dir: Output directory for results
        n_processes: Number of processes to use (default: CPU count)
        chunk_size: Size of chunks for processing (auto-calculated if None)
    
    Returns:
        Processing results and file paths
    """
    cleaner = MultiThreadedDatabaseCleaner(csv_path, json_rules_path, output_dir, n_processes, chunk_size)
    return cleaner.clean_database()


if __name__ == "__main__":
    # Example usage
    try:
        # Replace with your actual file paths
        csv_file = "Datasets/home-credit-default-risk/credit_card_balance.csv"
        rules_file = "cleaning_rules.json"
        output_directory = "output"
        
        # Use all available CPU cores
        n_processes = cpu_count()
        print(f"Using {n_processes} processes")
        
        results = clean_database_files(
            csv_file, 
            rules_file, 
            output_directory, 
            n_processes=n_processes
            # chunk_size will be auto-calculated for optimal performance
        )
        
        print("\nProcessing complete!")
        print("Summary:", results['summary'])
        print("Files saved:", results['files'])
        
    except Exception as e:
        print(f"Error: {e}")