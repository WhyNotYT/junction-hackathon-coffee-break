from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
import json
import numpy as np
import math
from pathlib import Path

app = Flask(__name__)

class DataSymphonyComposer:
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.csv_files = {}
        self.load_csv_files()
        
        # If no CSV files found, create demo data
        if not self.csv_files:
            print("🎵 No CSV files found, generating demo data...")
            self.create_demo_data()
        
        # Musical scales mapped to data patterns
        self.scales = {
            'happy': [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88],  # C Major
            'mysterious': [220.00, 246.94, 277.18, 329.63, 369.99, 415.30, 493.88],   # A Minor + chromatic
            'chaotic': [261.63, 277.18, 311.13, 369.99, 415.30, 466.16, 523.25], # Dissonant
            'peaceful': [261.63, 293.66, 349.23, 392.00, 440.00, 523.25]  # Pentatonic
        }
        
        self.instruments = ['piano', 'synth', 'bass', 'lead', 'pad', 'pluck', 'bell', 'drum']
    
    def create_demo_data(self):
        """Create demo data for testing when no CSV files are available"""
        demo_data = {
            'application_main': {
                'df': pd.DataFrame({
                    'SK_ID_CURR': [100001, 100002, 100003, 100004, 100005],
                    'NAME_CONTRACT_TYPE': ['Cash loans', 'Cash loans', 'Revolving loans', 'Cash loans', 'Cash loans'],
                    'CODE_GENDER': ['M', 'F', 'M', 'F', 'M'],
                    'FLAG_OWN_CAR': ['N', 'N', 'Y', 'Y', 'N'],
                    'FLAG_OWN_REALTY': ['Y', 'N', 'Y', 'Y', 'Y'],
                    'CNT_CHILDREN': [0, 0, 0, 0, 1],
                    'AMT_INCOME_TOTAL': [202500.0, 270000.0, 67500.0, 135000.0, 121500.0],
                    'AMT_CREDIT': [406597.5, 1293502.5, 135000.0, 312682.5, 513000.0],
                    'AMT_ANNUITY': [24700.5, 35698.5, 6750.0, 29686.5, 21865.5],
                    'NAME_INCOME_TYPE': ['Working', 'Commercial associate', 'Working', 'Working', 'Working'],
                    'NAME_EDUCATION_TYPE': ['Secondary / secondary special', 'Higher education', 'Secondary / secondary special', 'Secondary / secondary special', 'Secondary / secondary special'],
                    'NAME_FAMILY_STATUS': ['Single / not married', 'Single / not married', 'Married', 'Single / not married', 'Single / not married'],
                    'DAYS_BIRTH': [-9461, -16765, -19046, -19005, -19932],
                    'DAYS_EMPLOYED': [-637, -1188, -225, -3039, -3038],
                    'REGION_RATING_CLIENT': [2, 1, 2, 2, 2],
                    # Additional fields to make 28 total
                    'HOUR_APPR_PROCESS_START': [10, 11, 12, 13, 14],
                    'REG_REGION_NOT_LIVE_REGION': [0, 0, 0, 0, 0],
                    'REG_REGION_NOT_WORK_REGION': [0, 0, 0, 0, 0],
                    'LIVE_REGION_NOT_WORK_REGION': [0, 0, 0, 0, 0],
                    'REG_CITY_NOT_LIVE_CITY': [0, 0, 0, 0, 0],
                    'REG_CITY_NOT_WORK_CITY': [0, 0, 0, 0, 0],
                    'LIVE_CITY_NOT_WORK_CITY': [0, 0, 0, 0, 0],
                    'ORGANIZATION_TYPE': ['Business Entity Type 3', 'XNA', 'Self-employed', 'Business Entity Type 3', 'Other'],
                    'EXT_SOURCE_1': [0.083037, None, 0.311267, 0.622246, None],
                    'EXT_SOURCE_2': [0.262949, 0.622246, None, 0.555912, 0.650442],
                    'EXT_SOURCE_3': [0.139376, None, 0.731567, None, None],
                    'OBS_30_CNT_SOCIAL_CIRCLE': [2.0, 1.0, 0.0, 2.0, 0.0],
                    'DEF_30_CNT_SOCIAL_CIRCLE': [2.0, 0.0, 0.0, 0.0, 0.0],
                    'OBS_60_CNT_SOCIAL_CIRCLE': [1.0, 1.0, 0.0, 2.0, 0.0]
                }),
                'id_columns': ['SK_ID_CURR'],
                'columns': ['SK_ID_CURR', 'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'REGION_RATING_CLIENT', 'HOUR_APPR_PROCESS_START', 'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY', 'ORGANIZATION_TYPE', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE']
            },
            'bureau_balance': {
                'df': pd.DataFrame({
                    'SK_ID_BUREAU': [5001, 5002, 5003, 5004, 5005],
                    'SK_ID_CURR': [100001, 100002, 100003, 100004, 100005],
                    'MONTHS_BALANCE': [-1, -2, -3, -1, -4],
                    'STATUS': ['C', '0', 'X', 'C', '0'],
                    'CREDIT_ACTIVE': ['Active', 'Closed', 'Active', 'Closed', 'Active'],
                    'CREDIT_CURRENCY': ['currency 1', 'currency 1', 'currency 1', 'currency 1', 'currency 1'],
                    'DAYS_CREDIT': [-1000, -1500, -800, -1200, -2000],
                    'CREDIT_DAY_OVERDUE': [0, 0, 15, 0, 0],
                    'DAYS_CREDIT_ENDDATE': [-500, 0, -200, 0, -800],
                    'AMT_CREDIT_MAX_OVERDUE': [0, 0, 5000, 0, 0],
                    'CNT_CREDIT_PROLONG': [0, 0, 1, 0, 0],
                    'AMT_CREDIT_SUM': [45000, 0, 225000, 0, 450000],
                    'AMT_CREDIT_SUM_DEBT': [0, 0, 180000, 0, 0],
                    'AMT_CREDIT_SUM_LIMIT': [0, 0, 0, 0, 0],
                    'AMT_CREDIT_SUM_OVERDUE': [0, 0, 0, 0, 0]
                }),
                'id_columns': ['SK_ID_CURR', 'SK_ID_BUREAU'],
                'columns': ['SK_ID_BUREAU', 'SK_ID_CURR', 'MONTHS_BALANCE', 'STATUS', 'CREDIT_ACTIVE', 'CREDIT_CURRENCY', 'DAYS_CREDIT', 'CREDIT_DAY_OVERDUE', 'DAYS_CREDIT_ENDDATE', 'AMT_CREDIT_MAX_OVERDUE', 'CNT_CREDIT_PROLONG', 'AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM_LIMIT', 'AMT_CREDIT_SUM_OVERDUE']
            },
            'previous_application': {
                'df': pd.DataFrame({
                    'SK_ID_PREV': [2000001, 2000002, 2000003, 2000004, 2000005],
                    'SK_ID_CURR': [100001, 100002, 100003, 100004, 100005],
                    'NAME_CONTRACT_TYPE': ['Consumer loans', 'Cash loans', 'Consumer loans', 'Cash loans', 'Cash loans'],
                    'AMT_ANNUITY': [1420.5, 56553.0, 6737.5, 5357.25, 4813.2],
                    'AMT_APPLICATION': [24835.5, 179055.0, 20106.0, 47250.0, 67500.0],
                    'AMT_CREDIT': [23787.0, 179055.0, 19278.0, 45945.0, 675000.0],
                    'AMT_DOWN_PAYMENT': [2520.0, 0, 4230.0, 2835.0, 0],
                    'AMT_GOODS_PRICE': [24835.5, 179055.0, 20106.0, 47250.0, 675000.0],
                    'WEEKDAY_APPR_PROCESS_START': ['SATURDAY', 'THURSDAY', 'FRIDAY', 'MONDAY', 'WEDNESDAY'],
                    'FLAG_LAST_APPL_PER_CONTRACT': ['Y', 'Y', 'Y', 'Y', 'Y'],
                    'NAME_CASH_LOAN_PURPOSE': ['Repairs', 'XAP', 'Repairs', 'Repairs', 'XAP'],
                    'NAME_CONTRACT_STATUS': ['Approved', 'Approved', 'Approved', 'Refused', 'Approved'],
                    'DAYS_DECISION': [-1740, -606, -1186, -815, -617],
                    'NAME_PAYMENT_TYPE': ['XNA', 'Cash through the bank', 'XNA', 'XNA', 'Cash through the bank'],
                    'CODE_REJECT_REASON': ['XAP', 'XAP', 'XAP', 'HC', 'XAP']
                }),
                'id_columns': ['SK_ID_CURR', 'SK_ID_PREV'],
                'columns': ['SK_ID_PREV', 'SK_ID_CURR', 'NAME_CONTRACT_TYPE', 'AMT_ANNUITY', 'AMT_APPLICATION', 'AMT_CREDIT', 'AMT_DOWN_PAYMENT', 'AMT_GOODS_PRICE', 'WEEKDAY_APPR_PROCESS_START', 'FLAG_LAST_APPL_PER_CONTRACT', 'NAME_CASH_LOAN_PURPOSE', 'NAME_CONTRACT_STATUS', 'DAYS_DECISION', 'NAME_PAYMENT_TYPE', 'CODE_REJECT_REASON']
            }
        }
        
        self.csv_files = demo_data
        print(f"🎵 Created demo symphony with {len(demo_data)} instruments")
    
    def load_csv_files(self):
        if not os.path.exists(self.data_folder):
            return
        
        for file_path in Path(self.data_folder).glob("*.csv"):
            try:
                df = pd.read_csv(file_path, nrows=100)  # Limit for demo
                id_columns = [col for col in df.columns if any(x in col.upper() for x in ['ID', 'SK_', '_ID', 'KEY'])]
                if id_columns:
                    self.csv_files[file_path.stem] = {
                        'df': df,
                        'id_columns': id_columns,
                        'columns': list(df.columns)
                    }
                    print(f"🎵 Loaded musical data: {file_path.stem}")
            except Exception as e:
                print(f"⚠️ Skipped {file_path}: {e}")
    
    def compose_symphony(self, person_id):
        """Turn data relationships into musical composition"""
        instruments = []
        
        # Convert person_id to int if it's a string
        try:
            person_id = int(person_id)
        except ValueError:
            pass  # Keep as string if conversion fails
        
        for i, (table_name, file_info) in enumerate(self.csv_files.items()):
            df = file_info['df']
            id_columns = file_info['id_columns']
            
            # Find matching records
            person_data = None
            matched_column = None
            
            for id_col in id_columns:
                try:
                    # Check both int and string versions of person_id
                    if person_id in df[id_col].values or str(person_id) in df[id_col].astype(str).values:
                        person_data = df[df[id_col] == person_id].iloc[0] if person_id in df[id_col].values else df[df[id_col].astype(str) == str(person_id)].iloc[0]
                        matched_column = id_col
                        break
                except:
                    continue
            
            if person_data is not None:
                # Generate melody from data values - REMOVED THE LIMIT!
                melody = self.data_to_melody(person_data, table_name)
                
                # Assign instrument based on data characteristics
                instrument_type = self.instruments[i % len(self.instruments)]
                
                # Data-driven musical properties
                non_null_count = person_data.count()
                tempo = max(60, min(180, non_null_count * 2))
                
                # Scale based on data "mood"
                scale_type = self.analyze_data_mood(person_data, file_info['columns'])
                
                instruments.append({
                    'table_name': str(table_name),
                    'instrument': str(instrument_type),
                    'melody': melody,
                    'scale': [float(f) for f in self.scales[scale_type]],
                    'tempo': int(tempo),
                    'volume': float(min(0.8, non_null_count / 20)),
                    'matched_column': str(matched_column),
                    'data_density': int(non_null_count),
                    'total_columns': int(len(file_info['columns'])),
                    'column_data': self.extract_column_data(person_data, file_info['columns'])
                })
        
        return {
            'person_id': int(person_id) if str(person_id).isdigit() else str(person_id),
            'instruments': instruments,
            'composition_length': max(45, len(instruments) * 15),  # Dynamic length based on data
            'total_tables': len(instruments)
        }
    
    def data_to_melody(self, person_data, table_name):
        """Convert data row into musical notes - PLAYS ALL FIELDS!"""
        notes = []
        
        # Process ALL columns, not just the first 12
        for i, (col, value) in enumerate(person_data.items()):
            try:
                # Map different data types to frequencies
                if pd.isna(value):
                    # Even null values get a note (rest/quiet note)
                    frequency = 130.81  # Very low C for nulls
                    duration = 0.1  # Short duration for nulls
                    note_type = 'null'
                elif isinstance(value, (int, float)):
                    # Use modulo to map to scale degrees
                    scale_degree = int(abs(value)) % 7
                    frequency = 261.63 * (2 ** (scale_degree / 7))
                    duration = 0.2 + min((abs(value) % 10) / 20, 0.8)  # Variable duration
                    note_type = 'numeric'
                else:
                    # Hash strings to consistent notes
                    scale_degree = hash(str(value)) % 7
                    frequency = 261.63 * (2 ** (scale_degree / 7))
                    duration = 0.15 + min(len(str(value)) / 100, 0.6)  # String length affects duration
                    note_type = 'text'
                
                notes.append({
                    'frequency': frequency,
                    'duration': min(duration, 1.0),  # Max 1 second per note
                    'column': col,
                    'value': str(value)[:30] if pd.notna(value) else 'NULL',  # Truncate long values
                    'start_time': i * 0.3,  # Stagger note starts (slightly faster)
                    'note_type': note_type
                })
                    
            except Exception as e:
                # If there's an error processing a value, create a default note
                notes.append({
                    'frequency': 220.0,  # Default A note
                    'duration': 0.2,
                    'column': col,
                    'value': 'ERROR',
                    'start_time': i * 0.3,
                    'note_type': 'error'
                })
        
        print(f"🎵 {table_name}: Generated {len(notes)} notes from {len(person_data)} fields")
        return notes
    
    def analyze_data_mood(self, person_data, columns):
        """Determine musical mood based on data content"""
        risk_indicators = ['debt', 'overdue', 'default', 'late', 'reject', 'negative']
        positive_indicators = ['approve', 'good', 'high', 'income', 'positive', 'active']
        
        text_data = ' '.join([str(person_data[col]).lower() for col in columns if pd.notna(person_data[col])])
        
        risk_score = sum(1 for word in risk_indicators if word in text_data)
        positive_score = sum(1 for word in positive_indicators if word in text_data)
        
        if risk_score > positive_score + 1:
            return 'chaotic'
        elif positive_score > risk_score + 1:
            return 'happy'
        elif 'amount' in text_data or 'balance' in text_data:
            return 'mysterious'
        else:
            return 'peaceful'
    
    def extract_column_data(self, person_data, columns):
        """Extract column values for visual display"""
        column_data = []
        for col in columns:
            if col in person_data.index:
                value = person_data[col]
                # Convert numpy/pandas types to native Python types
                if pd.isna(value):
                    display_value = 'NULL'
                    has_data = False
                else:
                    # Convert to native Python types for JSON serialization
                    if isinstance(value, np.integer):
                        display_value = str(int(value))
                    elif isinstance(value, np.floating):
                        display_value = str(float(value))
                    else:
                        display_value = str(value)
                    has_data = True
                
                column_data.append({
                    'column': str(col),
                    'value': display_value[:30] if len(display_value) > 30 else display_value,
                    'has_data': has_data
                })
        return column_data

# Initialize composer
composer = DataSymphonyComposer('data')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/symphony/<person_id>')
def get_symphony(person_id):
    try:
        symphony_data = composer.compose_symphony(person_id)
        
        if not symphony_data['instruments']:
            return jsonify({'error': f'No data symphony found for person: {person_id}'})
        
        return jsonify(symphony_data)
        
    except Exception as e:
        return jsonify({'error': f'Symphony composition failed: {str(e)}'})

@app.route('/api/tables')
def get_available_tables():
    tables = list(composer.csv_files.keys())
    return jsonify({'tables': tables})

if __name__ == '__main__':
    print("🎼 Data Symphony Composer Starting...")
    print(f"🎵 Found {len(composer.csv_files)} musical data sources")
    app.run(debug=True)