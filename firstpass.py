import pandas as pd
import json
from groq import Groq
from typing import Dict, List, Tuple, Optional
import logging
import chardet

def detect_encoding(file_path: str, num_bytes: int = 100000) -> str:
    """Detect file encoding by reading a chunk of the file."""
    with open(file_path, "rb") as f:
        raw_data = f.read(num_bytes)  # read a chunk
    result = chardet.detect(raw_data)
    return result["encoding"]

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CSVDataAnalyzer:
    """
    Analyzes CSV data and generates cleaning rules using Groq API LLM.
    """
    
    def __init__(self, groq_api_key: str):
        """
        Initialize the analyzer with Groq API key.
        
        Args:
            groq_api_key (str): Your Groq API key
        """
        self.client = Groq(api_key=groq_api_key)
        self.model = "llama-3.1-8b-instant"  # Default Groq model
        
    def read_description_csv(self, description_file_path: str) -> pd.DataFrame:
        try:
            encoding = detect_encoding(description_file_path)
            logger.info(f"Detected encoding for description CSV: {encoding}")
            description_df = pd.read_csv(description_file_path, encoding=encoding)
            logger.info(f"Successfully loaded description CSV with {len(description_df)} rows")
            return description_df
        except Exception as e:
            logger.error(f"Error reading description CSV: {str(e)}")
            raise

    def filter_description_by_columns(self, description_df: pd.DataFrame, data_columns: List[str]) -> pd.DataFrame:
        """
        Filter description dataframe to only include rows that match the data columns.
        
        Args:
            description_df (pd.DataFrame): Full column descriptions dataframe
            data_columns (List[str]): List of column names from the data CSV
            
        Returns:
            pd.DataFrame: Filtered description dataframe containing only relevant columns
        """
        # Assume the first column of description_df contains the column names
        # You might need to adjust this based on your description CSV structure
        column_name_field = description_df.columns[0]  # Usually 'Table' or 'Column' or similar
        
        # Create a case-insensitive match for better matching
        data_columns_lower = [col.lower().strip() for col in data_columns]
        
        # Filter description rows that match data columns
        mask = description_df[column_name_field].astype(str).str.lower().str.strip().isin(data_columns_lower)
        filtered_df = description_df[mask].copy()
        
        # Log matching results
        matched_columns = filtered_df[column_name_field].tolist()
        unmatched_columns = [col for col in data_columns if col.lower().strip() not in 
                           [matched.lower().strip() for matched in matched_columns]]
        
        logger.info(f"Matched {len(filtered_df)} column descriptions out of {len(data_columns)} data columns")
        if unmatched_columns:
            logger.warning(f"No descriptions found for columns: {unmatched_columns}")
        
        return filtered_df
    
    def read_data_csv_sample(self, data_file_path: str, sample_rows: int = 3) -> Tuple[pd.DataFrame, List[str]]:
        """
        Read the data CSV file and extract headers and sample rows.
        
        Args:
            data_file_path (str): Path to the data CSV file
            sample_rows (int): Number of sample rows to extract (default: 3)
            
        Returns:
            Tuple[pd.DataFrame, List[str]]: Sample data and column headers
        """
        try:
            # Read the full CSV to get headers
            encoding = detect_encoding(data_file_path)
            logger.info(f"Detected encoding for data CSV: {encoding}")
            
            full_df = pd.read_csv(data_file_path, encoding=encoding)
            headers = full_df.columns.tolist()
            
            # Get sample rows
            sample_df = full_df.head(sample_rows)
            
            logger.info(f"Successfully loaded data CSV with {len(headers)} columns and {len(full_df)} total rows")
            logger.info(f"Extracted {len(sample_df)} sample rows")
            
            return sample_df, headers
        except Exception as e:
            logger.error(f"Error reading data CSV: {str(e)}")
            raise
    
    def create_analysis_prompt(self, filtered_description_df: pd.DataFrame, sample_df: pd.DataFrame, headers: List[str]) -> str:
        """
        Create the prompt for the LLM to analyze data and generate cleaning rules.
        
        Args:
            filtered_description_df (pd.DataFrame): Filtered column descriptions (only relevant columns)
            sample_df (pd.DataFrame): Sample data rows
            headers (List[str]): Column headers
            
        Returns:
            str: Formatted prompt for the LLM
        """
        
        # Convert filtered description dataframe to string
        if not filtered_description_df.empty:
            description_text = filtered_description_df.to_string(index=False)
        else:
            description_text = "No column descriptions available for the data columns."
        
        # Convert sample data to string
        sample_text = sample_df.to_string(index=False)
        
        # Create headers string
        headers_text = ", ".join(headers)
        
        prompt = f"""
You are a data quality expert. Analyze the provided CSV data and generate cleaning rules for each column.

RELEVANT COLUMN DESCRIPTIONS:
{description_text}

ALL COLUMN HEADERS:
{headers_text}

SAMPLE DATA (first 3 rows):
{sample_text}

Your task is to analyze each column and determine appropriate data cleaning rules. For each column, provide:
1. Lower limit (minimum acceptable value, use null if no minimum)
2. Upper limit (maximum acceptable value, use null if no maximum)  
3. Confidence score (0.1 to 1.0):
   - High confidence (0.8-1.0): Clear business logic violations (e.g., debt ratio > 1, negative age, percentage > 100)
   - Medium confidence (0.5-0.7): Reasonable but not absolute limits (e.g., income thresholds, typical age ranges)
   - Low confidence (0.1-0.4): Uncertain limits where domain expertise is needed

IMPORTANT RULES:
- For ratios/percentages: typically should be between 0 and 1 (or 0 and 100 if percentage format)
- For ages: typically should be positive and under reasonable human lifespan
- For monetary amounts: typically should be non-negative unless explicitly stated otherwise
- For counts/quantities: typically should be non-negative integers
- Use null for limits that don't make sense for the data type
- Be conservative with confidence scores - only use high confidence for obvious violations
- If no description is available for a column, infer the appropriate limits from the column name and sample data

OUTPUT FORMAT (JSON only, no other text):
{{
    "column_name_1": [lower_limit, upper_limit, confidence],
    "column_name_2": [lower_limit, upper_limit, confidence],
    ...
}}

Example output format:
{{
    "DebtRatio": [0, 1, 0.95],
    "Age": [0, 120, 0.85],
    "Income": [0, null, 0.3],
    "CreditScore": [300, 850, 0.9]
}}

Analyze ALL columns in the headers list and provide the JSON output:
"""
        return prompt
    
    def query_groq_api(self, prompt: str) -> str:
        """
        Send prompt to Groq API and get response.
        
        Args:
            prompt (str): The prompt to send to the LLM
            
        Returns:
            str: LLM response
        """
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=self.model,
                temperature=0.1,  # Low temperature for consistent results
                max_tokens=2048,
            )
            
            response = chat_completion.choices[0].message.content
            logger.info("Successfully received response from Groq API")
            return response
            
        except Exception as e:
            logger.error(f"Error querying Groq API: {str(e)}")
            raise
    
    def parse_llm_response(self, response: str) -> Dict:
        """
        Parse the LLM response and extract JSON.
        
        Args:
            response (str): Raw response from LLM
            
        Returns:
            Dict: Parsed cleaning rules
        """
        try:
            # Try to find JSON in the response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON found in response")
            
            json_str = response[start_idx:end_idx]
            cleaning_rules = json.loads(json_str)
            
            logger.info(f"Successfully parsed cleaning rules for {len(cleaning_rules)} columns")
            return cleaning_rules
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON from LLM response: {str(e)}")
            logger.error(f"Response was: {response}")
            raise
        except Exception as e:
            logger.error(f"Error processing LLM response: {str(e)}")
            raise
    
    def analyze_csv_data(self, description_file_path: str, data_file_path: str) -> Dict:
        """
        Main method to analyze CSV data and generate cleaning rules.
        
        Args:
            description_file_path (str): Path to column description CSV
            data_file_path (str): Path to data CSV file
            
        Returns:
            Dict: Cleaning rules in format {column_name: [lower_limit, upper_limit, confidence]}
        """
        logger.info("Starting CSV data analysis...")
        
        # Step 1: Read data CSV sample to get column names first
        sample_df, headers = self.read_data_csv_sample(data_file_path)
        
        # Step 2: Read description CSV
        description_df = self.read_description_csv(description_file_path)
        
        # Step 3: Filter description CSV to only relevant columns
        filtered_description_df = self.filter_description_by_columns(description_df, headers)
        
        # Step 4: Create analysis prompt with filtered descriptions
        prompt = self.create_analysis_prompt(filtered_description_df, sample_df, headers)
        
        # Step 5: Query Groq API
        response = self.query_groq_api(prompt)
        
        # Step 6: Parse response
        cleaning_rules = self.parse_llm_response(response)
        
        logger.info("CSV data analysis completed successfully")
        return cleaning_rules


def main():
    """
    Example usage of the CSVDataAnalyzer class.
    This function demonstrates how to use the analyzer.
    """
    # Replace with your actual Groq API key
    GROQ_API_KEY = "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"
    
    # Initialize analyzer
    analyzer = CSVDataAnalyzer(GROQ_API_KEY)
    
    try:
        # Analyze CSV data
        # Replace these with your actual file paths
        description_file = "Datasets/home-credit-default-risk/HomeCredit_columns_description.csv"
        data_file = "Datasets/home-credit-default-risk/credit_card_balance.csv"
        
        cleaning_rules = analyzer.analyze_csv_data(description_file, data_file)
        
        # Print results
        print("Generated Cleaning Rules:")
        print(json.dumps(cleaning_rules, indent=2))
        
        # Optionally save to file
        with open("cleaning_rules.json", "w") as f:
            json.dump(cleaning_rules, f, indent=2)
        
        print("Cleaning rules saved to cleaning_rules.json")
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")


if __name__ == "__main__":
    main()