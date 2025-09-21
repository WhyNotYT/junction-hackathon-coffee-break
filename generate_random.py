import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Set random seed for reproducible demo data
np.random.seed(42)
random.seed(42)

def generate_application_main():
    """Generate main application data - the core quantum particle"""
    n_records = 500
    
    # Generate SK_ID_CURR values
    sk_ids = [100000 + i for i in range(n_records)]
    
    data = {
        'SK_ID_CURR': sk_ids,
        'TARGET': np.random.choice([0, 1], n_records, p=[0.8, 0.2]),  # 20% default rate
        'NAME_CONTRACT_TYPE': np.random.choice(['Cash loans', 'Revolving loans'], n_records, p=[0.9, 0.1]),
        'CODE_GENDER': np.random.choice(['M', 'F'], n_records, p=[0.6, 0.4]),
        'FLAG_OWN_CAR': np.random.choice(['Y', 'N'], n_records, p=[0.4, 0.6]),
        'FLAG_OWN_REALTY': np.random.choice(['Y', 'N'], n_records, p=[0.7, 0.3]),
        'CNT_CHILDREN': np.random.choice([0, 1, 2, 3, 4, 5], n_records, p=[0.4, 0.3, 0.2, 0.05, 0.03, 0.02]),
        'AMT_INCOME_TOTAL': np.random.lognormal(10.5, 0.8, n_records).astype(int),
        'AMT_CREDIT': np.random.lognormal(11.8, 0.6, n_records).astype(int),
        'AMT_ANNUITY': np.random.normal(25000, 15000, n_records).clip(5000, None).astype(int),
        'AMT_GOODS_PRICE': np.random.lognormal(11.5, 0.7, n_records).astype(int),
        'NAME_INCOME_TYPE': np.random.choice([
            'Working', 'Commercial associate', 'Pensioner', 'State servant', 'Student'
        ], n_records, p=[0.5, 0.2, 0.15, 0.1, 0.05]),
        'NAME_EDUCATION_TYPE': np.random.choice([
            'Secondary / secondary special', 'Higher education', 'Incomplete higher', 'Lower secondary'
        ], n_records, p=[0.6, 0.25, 0.1, 0.05]),
        'NAME_FAMILY_STATUS': np.random.choice([
            'Married', 'Single / not married', 'Civil marriage', 'Separated', 'Widow'
        ], n_records, p=[0.4, 0.3, 0.15, 0.1, 0.05]),
        'NAME_HOUSING_TYPE': np.random.choice([
            'House / apartment', 'With parents', 'Municipal apartment', 'Rented apartment', 'Office apartment'
        ], n_records, p=[0.5, 0.2, 0.15, 0.1, 0.05]),
        'DAYS_BIRTH': -np.random.randint(18*365, 70*365, n_records),  # Age in days (negative)
        'DAYS_EMPLOYED': -np.random.randint(0, 40*365, n_records),     # Employment in days (negative)
        'FLAG_MOBIL': np.random.choice([0, 1], n_records, p=[0.1, 0.9]),
        'FLAG_EMP_PHONE': np.random.choice([0, 1], n_records, p=[0.2, 0.8]),
        'FLAG_WORK_PHONE': np.random.choice([0, 1], n_records, p=[0.3, 0.7]),
        'FLAG_CONT_MOBILE': np.random.choice([0, 1], n_records, p=[0.05, 0.95]),
        'FLAG_PHONE': np.random.choice([0, 1], n_records, p=[0.4, 0.6]),
        'FLAG_EMAIL': np.random.choice([0, 1], n_records, p=[0.3, 0.7]),
        'REGION_RATING_CLIENT': np.random.choice([1, 2, 3], n_records, p=[0.3, 0.5, 0.2]),
        'HOUR_APPR_PROCESS_START': np.random.randint(0, 24, n_records),
        'REG_REGION_NOT_LIVE_REGION': np.random.choice([0, 1], n_records, p=[0.9, 0.1]),
        'REG_REGION_NOT_WORK_REGION': np.random.choice([0, 1], n_records, p=[0.8, 0.2]),
        'LIVE_REGION_NOT_WORK_REGION': np.random.choice([0, 1], n_records, p=[0.85, 0.15]),
    }
    
    df = pd.DataFrame(data)
    df.to_csv('data/application_main.csv', index=False)
    print("⚛️ Created application_main.csv - Core quantum particle data")
    return df

def generate_bureau_data():
    """Generate credit bureau data - external quantum field"""
    # Generate records for about 70% of application IDs
    app_ids = [100000 + i for i in range(500)]
    bureau_ids = random.sample(app_ids, 350)  # 70% coverage
    
    bureau_records = []
    for sk_id in bureau_ids:
        # Each person can have multiple bureau records
        n_records = np.random.poisson(2) + 1  # 1-5 records per person
        
        for i in range(n_records):
            record = {
                'SK_ID_CURR': sk_id,
                'SK_ID_BUREAU': sk_id * 1000 + i,
                'CREDIT_ACTIVE': np.random.choice(['Active', 'Closed', 'Bad debt', 'Sold'], p=[0.4, 0.5, 0.05, 0.05]),
                'CREDIT_CURRENCY': 'currency 1',  # Simplified
                'DAYS_CREDIT': -np.random.randint(0, 10*365),  # Days before application
                'CREDIT_DAY_OVERDUE': max(0, np.random.normal(0, 50)),
                'DAYS_CREDIT_ENDDATE': np.random.randint(-365*5, 365*10),  # End date
                'DAYS_ENDDATE_FACT': np.random.randint(-365*5, 365*10),    # Actual end date
                'AMT_CREDIT_MAX_OVERDUE': max(0, np.random.exponential(50000)),
                'CNT_CREDIT_PROLONG': np.random.poisson(0.5),
                'AMT_CREDIT_SUM': int(np.random.lognormal(11, 1)),
                'AMT_CREDIT_SUM_DEBT': int(np.random.lognormal(10, 1.5)),
                'AMT_CREDIT_SUM_LIMIT': int(np.random.lognormal(11.5, 1.2)),
                'AMT_CREDIT_SUM_OVERDUE': max(0, np.random.exponential(10000)),
                'CREDIT_TYPE': np.random.choice([
                    'Consumer credit', 'Credit card', 'Mortgage', 'Car loan', 'Microloan'
                ], p=[0.3, 0.25, 0.2, 0.15, 0.1]),
                'DAYS_CREDIT_UPDATE': -np.random.randint(0, 365),
                'AMT_ANNUITY': int(np.random.lognormal(9, 1)),
            }
            bureau_records.append(record)
    
    df = pd.DataFrame(bureau_records)
    df.to_csv('data/bureau.csv', index=False)
    print("🔬 Created bureau.csv - Credit bureau quantum entanglement data")
    return df

def generate_previous_application():
    """Generate previous application data - historical quantum states"""
    # Generate for about 60% of main application IDs
    app_ids = [100000 + i for i in range(500)]
    prev_app_ids = random.sample(app_ids, 300)
    
    prev_records = []
    for sk_id in prev_app_ids:
        # Each person can have multiple previous applications
        n_prev = np.random.poisson(1.5) + 1  # 1-4 previous applications
        
        for i in range(n_prev):
            record = {
                'SK_ID_CURR': sk_id,
                'SK_ID_PREV': sk_id * 100 + i,
                'NAME_CONTRACT_TYPE': np.random.choice(['Consumer loans', 'Cash loans', 'Revolving loans'], p=[0.5, 0.4, 0.1]),
                'AMT_ANNUITY': int(np.random.lognormal(10, 0.8)),
                'AMT_APPLICATION': int(np.random.lognormal(12, 0.7)),
                'AMT_CREDIT': int(np.random.lognormal(11.5, 0.8)),
                'AMT_DOWN_PAYMENT': int(np.random.exponential(50000)),
                'AMT_GOODS_PRICE': int(np.random.lognormal(11.8, 0.9)),
                'WEEKDAY_APPR_PROCESS_START': np.random.choice(['MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY', 'SATURDAY', 'SUNDAY']),
                'HOUR_APPR_PROCESS_START': np.random.randint(0, 24),
                'FLAG_LAST_APPL_PER_CONTRACT': np.random.choice(['Y', 'N'], p=[0.8, 0.2]),
                'NAME_CASH_LOAN_PURPOSE': np.random.choice([
                    'XAP', 'XNA', 'Repairs', 'Other', 'Education', 'Furniture', 'Business development'
                ], p=[0.3, 0.2, 0.15, 0.1, 0.1, 0.1, 0.05]),
                'NAME_CONTRACT_STATUS': np.random.choice(['Approved', 'Canceled', 'Refused', 'Unused offer'], p=[0.6, 0.2, 0.15, 0.05]),
                'DAYS_DECISION': -np.random.randint(30, 365*5),
                'NAME_PAYMENT_TYPE': np.random.choice(['Cash through the bank', 'XNA', 'Non-cash from your account'], p=[0.4, 0.4, 0.2]),
                'CODE_REJECT_REASON': np.random.choice(['XAP', 'HC', 'LIMIT', 'SCO', 'CLIENT', 'SYSTEM', 'XNA'], p=[0.7, 0.1, 0.05, 0.05, 0.05, 0.03, 0.02]),
                'NAME_TYPE_SUITE': np.random.choice(['Unaccompanied', 'With spouse', 'Family', 'Other_B', 'Other_A', 'Group of people'], p=[0.6, 0.15, 0.1, 0.05, 0.05, 0.05]),
                'NAME_CLIENT_TYPE': np.random.choice(['Repeater', 'New', 'Refreshed', 'XNA'], p=[0.4, 0.35, 0.2, 0.05]),
                'NAME_GOODS_CATEGORY': np.random.choice(['XNA', 'Mobile', 'Computers', 'Photo / Cinema Equipment', 'Consumer Electronics', 'Furniture'], p=[0.4, 0.2, 0.15, 0.1, 0.1, 0.05]),
                'NAME_PORTFOLIO': np.random.choice(['POS', 'Cash', 'Cards', 'XNA'], p=[0.4, 0.3, 0.25, 0.05]),
                'NAME_PRODUCT_TYPE': np.random.choice(['XNA', 'walk-in', 'x-sell'], p=[0.6, 0.3, 0.1]),
                'CHANNEL_TYPE': np.random.choice(['Credit and cash offices', 'Country-wide', 'Stone', 'Regional / Local', 'Contact center', 'Car dealer'], p=[0.3, 0.25, 0.2, 0.1, 0.1, 0.05]),
            }
            prev_records.append(record)
    
    df = pd.DataFrame(prev_records)
    df.to_csv('data/previous_application.csv', index=False)
    print("🌌 Created previous_application.csv - Historical quantum states")
    return df

def generate_credit_card_balance():
    """Generate credit card balance data - dynamic quantum oscillations"""
    # Generate for credit card holders (about 40% of population)
    app_ids = [100000 + i for i in range(500)]
    cc_holders = random.sample(app_ids, 200)
    
    cc_records = []
    for sk_id in cc_holders:
        # Multiple months of credit card data per person
        n_months = np.random.randint(6, 36)  # 6 months to 3 years of data
        
        for month in range(n_months):
            record = {
                'SK_ID_CURR': sk_id,
                'SK_ID_PREV': sk_id * 10 + (month % 10),  # Link to previous application
                'MONTHS_BALANCE': -(month + 1),  # Months before application
                'AMT_BALANCE': max(0, np.random.normal(50000, 30000)),
                'AMT_CREDIT_LIMIT_ACTUAL': np.random.uniform(50000, 500000),
                'AMT_DRAWINGS_ATM_CURRENT': max(0, np.random.exponential(5000)),
                'AMT_DRAWINGS_CURRENT': max(0, np.random.exponential(10000)),
                'AMT_DRAWINGS_OTHER_CURRENT': max(0, np.random.exponential(3000)),
                'AMT_DRAWINGS_POS_CURRENT': max(0, np.random.exponential(8000)),
                'AMT_INST_MIN_REGULARITY': max(0, np.random.normal(2000, 1000)),
                'AMT_PAYMENT_CURRENT': max(0, np.random.normal(15000, 10000)),
                'AMT_PAYMENT_TOTAL_CURRENT': max(0, np.random.normal(18000, 12000)),
                'AMT_RECEIVABLE_PRINCIPAL': max(0, np.random.normal(30000, 20000)),
                'AMT_RECIVABLE': max(0, np.random.normal(32000, 22000)),
                'AMT_TOTAL_RECEIVABLE': max(0, np.random.normal(35000, 25000)),
                'CNT_DRAWINGS_ATM_CURRENT': max(0, int(np.random.poisson(2))),
                'CNT_DRAWINGS_CURRENT': max(0, int(np.random.poisson(5))),
                'CNT_DRAWINGS_OTHER_CURRENT': max(0, int(np.random.poisson(1))),
                'CNT_DRAWINGS_POS_CURRENT': max(0, int(np.random.poisson(8))),
                'CNT_INSTALMENT_MATURE_CUM': max(0, int(np.random.poisson(month + 1))),
                'NAME_CONTRACT_STATUS': np.random.choice(['Active', 'Completed', 'Demand', 'Refused', 'Approved', 'Cancelled', 'Signed'], p=[0.6, 0.15, 0.1, 0.05, 0.05, 0.03, 0.02]),
                'SK_DPD': max(0, int(np.random.exponential(2))),  # Days past due
                'SK_DPD_DEF': max(0, int(np.random.exponential(1))),  # Days past due with tolerance
            }
            cc_records.append(record)
    
    df = pd.DataFrame(cc_records)
    df.to_csv('data/credit_card_balance.csv', index=False)
    print("💳 Created credit_card_balance.csv - Credit card quantum oscillations")
    return df

def generate_installments_payments():
    """Generate installment payments data - payment quantum dynamics"""
    # Generate for about 80% of population
    app_ids = [100000 + i for i in range(500)]
    installment_holders = random.sample(app_ids, 400)
    
    payment_records = []
    for sk_id in installment_holders:
        # Multiple installment records per person
        n_payments = np.random.randint(12, 120)  # 1-10 years of payments
        
        for payment_num in range(n_payments):
            # Calculate payment date
            days_before = np.random.randint(30 * payment_num, 30 * (payment_num + 1))
            
            amt_instalment = np.random.normal(5000, 2000)
            amt_payment = amt_instalment + np.random.normal(0, 500)  # Sometimes over/under pay
            
            record = {
                'SK_ID_CURR': sk_id,
                'SK_ID_PREV': sk_id * 10 + (payment_num % 10),
                'NUM_INSTALMENT_VERSION': np.random.choice([0, 1, 2], p=[0.8, 0.15, 0.05]),
                'NUM_INSTALMENT_NUMBER': payment_num + 1,
                'DAYS_INSTALMENT': -days_before,
                'DAYS_ENTRY_PAYMENT': -days_before + np.random.randint(-5, 15),  # Payment might be late/early
                'AMT_INSTALMENT': max(0, amt_instalment),
                'AMT_PAYMENT': max(0, amt_payment),
            }
            payment_records.append(record)
    
    df = pd.DataFrame(payment_records)
    df.to_csv('data/installments_payments.csv', index=False)
    print("💰 Created installments_payments.csv - Payment quantum dynamics")
    return df

def generate_pos_cash_balance():
    """Generate POS and cash balance data - retail quantum interactions"""
    # Generate for about 50% of population
    app_ids = [100000 + i for i in range(500)]
    pos_holders = random.sample(app_ids, 250)
    
    pos_records = []
    for sk_id in pos_holders:
        # Multiple months of POS data
        n_months = np.random.randint(6, 48)
        
        for month in range(n_months):
            record = {
                'SK_ID_CURR': sk_id,
                'SK_ID_PREV': sk_id * 10 + (month % 10),
                'MONTHS_BALANCE': -(month + 1),
                'CNT_INSTALMENT': np.random.randint(6, 60),  # 6 months to 5 years
                'CNT_INSTALMENT_FUTURE': max(0, np.random.randint(0, 50)),
                'NAME_CONTRACT_STATUS': np.random.choice(['Active', 'Completed', 'Canceled', 'Returned to the store', 'Approved', 'Demand', 'Signed', 'Amortized debt', 'XNA'], p=[0.5, 0.25, 0.1, 0.05, 0.03, 0.03, 0.02, 0.01, 0.01]),
                'SK_DPD': max(0, int(np.random.exponential(3))),
                'SK_DPD_DEF': max(0, int(np.random.exponential(1.5))),
            }
            pos_records.append(record)
    
    df = pd.DataFrame(pos_records)
    df.to_csv('data/POS_CASH_balance.csv', index=False)
    print("🏪 Created POS_CASH_balance.csv - Retail quantum interactions")
    return df

def main():
    """Generate all demo CSV files"""
    import os
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    print("🚀 Generating Quantum Credit Entanglement Demo Data...")
    print("=" * 60)
    
    # Generate all data files
    app_df = generate_application_main()
    bureau_df = generate_bureau_data()
    prev_app_df = generate_previous_application()
    cc_df = generate_credit_card_balance()
    install_df = generate_installments_payments()
    pos_df = generate_pos_cash_balance()
    
    print("=" * 60)
    print(f"✨ Demo data generated successfully!")
    print(f"📊 Total files: 6")
    print(f"👥 Main applications: {len(app_df)}")
    print(f"📋 Bureau records: {len(bureau_df)}")
    print(f"📑 Previous applications: {len(prev_app_df)}")
    print(f"💳 Credit card records: {len(cc_df)}")
    print(f"💰 Payment records: {len(install_df)}")
    print(f"🏪 POS records: {len(pos_df)}")
    
    print("\n🔬 Try these SK_ID_CURR values for quantum entanglement:")
    sample_ids = random.sample([100000 + i for i in range(500)], 10)
    for sk_id in sample_ids:
        print(f"   ⚛️ {sk_id}")
    
    print("\n🌌 Ready for quantum credit risk analysis!")

if __name__ == "__main__":
    main()